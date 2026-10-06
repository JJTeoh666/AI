import unittest
from unittest.mock import patch

import numpy as np

from piano_ml.calibration import Calibration
from piano_ml.decode import DecodeCache, decode_outputs, find_peaks
from piano_ml.metrics import PreparedNoteMatcher, note_counts_by_pitch, window_note_pairs
from piano_ml.thresholds import Thresholds, threshold_grid


class CalibrationSpeedTest(unittest.TestCase):
    def test_cached_decoder_exactly_matches_original_events_and_details(self):
        random = np.random.default_rng(19)
        grids = threshold_grid(Thresholds(), [0.35, 0.5, 0.8], [0.35, 0.5, 0.8], [0.35, 0.5, 0.8])
        grids.append(Thresholds(**{head: tuple(random.choice([0.35, 0.5, 0.8], 88))
                                   for head in ("frame", "onset", "offset")}))
        for length, releases in ((1, 3), (3, 5), (51, 2), (51, 3), (201, 3)):
            outputs = {head: random.choice([0, 0.2, 0.35, 0.5, 0.65, 0.9, 1], (length, 88)).astype(np.float32)
                       for head in ("frame", "onset", "offset", "velocity")}
            outputs["pedal"] = random.random((length, 1)).astype(np.float32)
            cache = DecodeCache(outputs)
            for values in grids:
                options = dict(duration=length * 0.02, threshold=values.frame,
                               onset_threshold=values.onset, offset_threshold=values.offset,
                               release_frames=releases, include_chords=False)
                original = decode_outputs(outputs, **options)
                cached = decode_outputs(outputs, _cache=cache, **options)
                self.assertEqual(cached, original, (length, releases, values))
            # The same cache also supports models without an offset head.
            outputs.pop("offset")
            cache = DecodeCache(outputs)
            self.assertEqual(decode_outputs(outputs, length * 0.02, 0.5, release_frames=releases),
                             decode_outputs(outputs, length * 0.02, 0.5, release_frames=releases, _cache=cache))

    def test_offset_wins_when_it_coincides_with_gap_confirmation(self):
        outputs = {head: np.zeros((20, 88), np.float32) for head in ("frame", "onset", "offset")}
        outputs["frame"][1:5, 3] = 0.9
        outputs["onset"][1, 3] = 0.9
        outputs["offset"][7, 3] = 0.9
        options = dict(duration=0.4, threshold=0.5, release_frames=3, include_chords=False)
        result = decode_outputs(outputs, _cache=DecodeCache(outputs), **options)
        self.assertEqual(result, decode_outputs(outputs, **options))
        self.assertEqual(result["notes"][0]["end"], 0.14)

    def test_prepared_matching_preserves_unique_counts_and_boundary_rules(self):
        reference = [dict(pitch=24, start=0.2, end=0.6), dict(pitch=24, start=0.22, end=0.7),
                     dict(pitch=28, start=-0.1, end=0.5), dict(pitch=33, start=0.7, end=1.3),
                     dict(pitch=60, start=0.1, end=0.5)]
        guesses = [dict(pitch=24, start=0.21, end=0.6), dict(pitch=24, start=0.22, end=0.95),
                   dict(pitch=28, start=0.01, end=0.5), dict(pitch=33, start=0.7, end=1.0),
                   dict(pitch=60, start=0.1, end=0.5), dict(pitch=61, start=0.3, end=1.0)]
        for window in (False, True):
            matcher = PreparedNoteMatcher(reference, 1.0 if window else None)
            actual = matcher.counts(guesses)
            refs, notes = window_note_pairs(reference, guesses, 1.0) if window else (reference, guesses)
            self.assertEqual({p: pair[0] for p, pair in actual.items()}, note_counts_by_pitch(refs, notes, offsets=False))
            self.assertEqual({p: pair[1] for p, pair in actual.items()}, note_counts_by_pitch(refs, notes, offsets=True))

    def test_grid_reuses_peaks_and_returns_identical_all_candidate_metrics(self):
        outputs = {head: np.zeros((60, 88), np.float32) for head in ("frame", "onset", "offset")}
        outputs["frame"][10:35, 3] = 0.7
        outputs["frame"][25:27, 3] = 0.1
        outputs["onset"][10, 3] = 0.7
        outputs["offset"][35, 3] = 0.6
        reference = [dict(pitch=24, start=0.2, end=0.7)]
        truth = outputs["frame"] > 0.5
        grid = threshold_grid(Thresholds(), [0.35, 0.5, 0.65, 0.8], [0.35, 0.5, 0.65, 0.8], [0.35, 0.5, 0.65, 0.8])
        calibration = Calibration(grid, release_frames=3)
        with patch("piano_ml.decode.find_peaks", wraps=find_peaks) as peaks:
            calibration.add(outputs, truth, reference, 1.2, window=True)
        self.assertEqual(peaks.call_count, 8 * 88)  # Four onset and four offset values.
        for values in grid:
            notes = decode_outputs(outputs, 1.2, values.frame, include_chords=False,
                onset_threshold=values.onset, offset_threshold=values.offset, release_frames=3)["notes"]
            refs, notes = window_note_pairs(reference, notes, 1.2)
            expected = note_counts_by_pitch(refs, notes)
            actual = calibration.pitch_counts[values]
            for pitch in range(21, 109):
                self.assertEqual(tuple(actual[pitch - 21]), expected.get(pitch, (0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
