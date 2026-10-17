"""Already sustained V11 notes must not receive a second pedal extension."""

import copy
import tempfile
import unittest
from pathlib import Path

from piano_ml.soundfont import _events
from piano_ml.synthesis import render_result_wav


class SustainPlaybackTest(unittest.TestCase):
    def result(self, marked=True):
        result = {"duration": 1., "notes": [
            {"pitch": 60, "start": .1, "end": .4, "velocity": 80}],
            "pedals": [{"start": .2, "end": .9}]}
        if marked:
            result["note_duration"] = "sustain_extended"
        return result

    def test_sampled_events_keep_predicted_end_and_only_final_pedal_reset(self):
        result = self.result()
        original = copy.deepcopy(result)
        self.assertEqual(_events(result, 16000), [
            (1600, 3, "on", 60, 80), (6400, 2, "off", 60, 0), (16000, 4, "cc", 64, 0)])
        self.assertEqual(result, original)

    def test_repeated_strikes_release_previous_key_before_next_onset(self):
        result = self.result()
        result["notes"][0]["end"] = .8
        result["notes"].append({"pitch": 60, "start": .5, "end": .7, "velocity": 100})
        self.assertEqual(_events(result, 16000), [
            (1600, 3, "on", 60, 80), (8000, 2, "off", 60, 0),
            (8000, 3, "on", 60, 100), (11200, 2, "off", 60, 0), (16000, 4, "cc", 64, 0)])

    def test_basic_audio_ignores_extra_pedal_hold_only_for_marked_notes(self):
        result = self.result()
        original = copy.deepcopy(result)
        dry = {**result, "pedals": []}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            sustained = render_result_wav(result, root / "sustained.wav", engine="basic")
            dry_audio = render_result_wav(dry, root / "dry.wav", engine="basic")
            self.assertEqual(sustained.read_bytes(), dry_audio.read_bytes())
            legacy_audio = render_result_wav(self.result(marked=False), root / "legacy.wav", engine="basic")
            self.assertNotEqual(legacy_audio.read_bytes(), dry_audio.read_bytes())
        self.assertEqual(result, original)

    def test_unmarked_and_explicit_key_release_results_retain_pedal_events(self):
        result = self.result(marked=False)
        # ByteDance results do not declare this V11 duration marker.
        result["engine"] = "bytedance"
        expected = [(3200, 1, "cc", 64, 127), (14400, 0, "cc", 64, 0), (16000, 4, "cc", 64, 0)]
        self.assertEqual([event for event in _events(result, 16000) if event[2] == "cc"], expected)
        result["note_duration"] = "key_release"
        self.assertEqual([event for event in _events(result, 16000) if event[2] == "cc"], expected)

    def test_marked_results_still_validate_pedal_intervals(self):
        result = self.result()
        result["pedals"] = [{"start": .2, "end": 1.1}]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            for engine in ("basic", "sampled"):
                with self.subTest(engine=engine), self.assertRaisesRegex(ValueError, "pedal timing"):
                    render_result_wav(result, Path(directory) / "invalid.wav", engine=engine)


if __name__ == "__main__":
    unittest.main()
