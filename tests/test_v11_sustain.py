"""Sustain-extended labels and independent pedal event decoding for V11."""

import unittest
from unittest.mock import patch

import numpy as np

from piano_ml.data import MaestroWindows, collate_windows
from piano_ml.decode import decode_outputs, decode_pedal_events
from piano_ml.midi import Note, Pedal, extend_sustain_notes, merge_pedal_intervals


class SustainNotesTest(unittest.TestCase):
    def test_pedal_union_boundaries_repeated_strikes_and_no_mutation(self):
        notes = [Note(60, .1, .4, 90), Note(60, 1.4, 1.5, 100),
                 Note(62, .1, .3), Note(64, .1, 2.), Note(65, .1, .2)]
        pedals = [Pedal(.3, .8), Pedal(.7, 1.2), Pedal(1.2, 2.), Pedal(3, 3)]
        originals = list(notes), list(pedals)
        result = extend_sustain_notes(notes, pedals)
        self.assertEqual(merge_pedal_intervals(pedals), [Pedal(.3, 2.)])
        self.assertEqual([(note.pitch, note.end) for note in result],
                         [(60, 1.4), (62, 2.), (64, 2.), (65, .2), (60, 2.)])
        self.assertEqual((notes, pedals), originals)
        self.assertTrue(all(result_note is not original for result_note in result for original in notes))
        self.assertEqual(result[0].velocity, 90)
        self.assertEqual(extend_sustain_notes(notes, []), sorted(notes, key=lambda note: (note.start, note.pitch)))

    def dataset(self, notes, pedals, **kwargs):
        row = {"audio_filename": "audio.wav", "midi_filename": "audio.midi",
               "duration": str(kwargs.pop("duration", 1.))}
        with patch("piano_ml.data.load_records", return_value=[row]), \
                patch("piano_ml.data.read_performance", return_value=(notes, pedals)):
            dataset = MaestroWindows(".", "train", seconds=1, windows_per_file=kwargs.pop("windows_per_file", 1),
                                     multi_target=True, **kwargs)
        dataset._performances["audio.midi"] = notes, pedals
        return dataset

    @patch("piano_ml.data.read_wav_window", return_value=np.zeros(16000, np.float32))
    def test_extended_frames_offsets_reference_and_two_frame_pedal_targets(self, read_audio):
        notes, pedals = [Note(60, .1, .3, 80)], [Pedal(.2, .7)]
        ordinary = self.dataset(notes, pedals)
        extended = self.dataset(notes, pedals, note_duration="sustain_extended", pedal_events=True)
        _, old = ordinary[0]
        _, new = extended[0]
        self.assertEqual(old["reference_notes"][0]["end"], .3)
        self.assertEqual(new["reference_notes"][0]["end"], .7)
        self.assertEqual(new["frame"][25, 60 - 21], 1)
        self.assertEqual(old["frame"][25, 60 - 21], 0)
        np.testing.assert_array_equal(np.flatnonzero(new["offset"][:, 60 - 21]), [35, 36])
        np.testing.assert_array_equal(np.flatnonzero(new["pedal_onset"][:, 0]), [10, 11])
        np.testing.assert_array_equal(np.flatnonzero(new["pedal_offset"][:, 0]), [35, 36])
        self.assertNotIn("pedal_onset", old)
        self.assertNotIn("pedal_offset", old)
        self.assertEqual(new["pedal_onset"].shape, (51, 1))
        _, batch = collate_windows([extended[0], extended[0]])
        self.assertEqual(tuple(batch["pedal_offset"].shape), (2, 51, 1))

    @patch("piano_ml.data.read_wav_window", return_value=np.zeros(16000, np.float32))
    def test_full_performance_carry_in_and_offsets_across_windows(self, read_audio):
        dataset = self.dataset([Note(60, .1, .2)], [Pedal(0, 2.2)], duration=3., windows_per_file=2,
                               note_duration="sustain_extended", pedal_events=True)
        _, first = dataset[0]  # 0.5--1.5 seconds
        _, second = dataset[1]  # 1.5--2.5 seconds
        self.assertAlmostEqual(first["reference_notes"][0]["start"], -.4)
        self.assertAlmostEqual(first["reference_notes"][0]["end"], 1.7)
        self.assertTrue(np.all(first["frame"][:, 60 - 21] == 1))
        self.assertEqual(first["pedal_onset"].sum(), 0)
        self.assertEqual(first["pedal_offset"].sum(), 0)
        self.assertEqual(second["pedal_onset"].sum(), 0)
        np.testing.assert_array_equal(np.flatnonzero(second["pedal_offset"][:, 0]), [35, 36])
        self.assertEqual(second["frame"][34, 60 - 21], 1)
        self.assertEqual(second["frame"][36, 60 - 21], 0)

    @patch("piano_ml.data.read_wav_window", return_value=np.zeros(16000, np.float32))
    def test_focused_preloaded_cache_remains_raw_and_transforms_once(self, read_audio):
        notes, pedals = [Note(30, .1, .2)], [Pedal(0, 1.5)]
        dataset = self.dataset(notes, pedals, duration=2., random_windows=True, bass_sampling=1.,
                               note_duration="sustain_extended", pedal_events=True)
        with patch("piano_ml.data.random.uniform", return_value=.5), \
                patch("piano_ml.data.extend_sustain_notes", wraps=extend_sustain_notes) as transform:
            _, first = dataset[0]
            _, second = dataset[0]
        self.assertEqual(transform.call_count, 1)
        self.assertEqual(dataset._performances["audio.midi"][0][0].end, .2)
        self.assertEqual(dataset.bass_anchors[0][1].end, .2)
        self.assertEqual(first["reference_notes"], second["reference_notes"])
        self.assertEqual(first["reference_notes"][0]["end"], 1.)
        with self.assertRaisesRegex(ValueError, "Note duration"):
            self.dataset(notes, pedals, note_duration="unknown")


class PedalEventsTest(unittest.TestCase):
    def probabilities(self, frames=101):
        return {"frame": np.zeros((frames, 88), np.float32),
                "onset": np.zeros((frames, 88), np.float32),
                "offset": np.zeros((frames, 88), np.float32),
                "velocity": np.zeros((frames, 88), np.float32),
                "pedal": np.zeros((frames, 1), np.float32)}

    def test_event_guidance_release_and_next_press(self):
        output = self.probabilities()
        output["pedal"][10:90] = .9
        output["pedal_onset"] = np.zeros((101, 1), np.float32)
        output["pedal_offset"] = np.zeros((101, 1), np.float32)
        output["pedal_onset"][[9, 60]] = .8
        output["pedal_offset"][[40, 80]] = .8
        result = decode_outputs(output, 2., .5)
        self.assertEqual(result["pedals"], [{"start": .18, "end": .8}, {"start": 1.2, "end": 1.6}])
        self.assertEqual(result["notes"], [])

    def test_activity_fallback_carry_in_short_dips_and_clip_closure(self):
        output = self.probabilities(101)
        output["pedal"][:30] = .9
        output["pedal"][15:18] = 0  # A 60 ms dip does not release the pedal.
        output["pedal"][60:] = .9
        no_events = np.zeros((101, 1), np.float32)
        result = decode_pedal_events(output["pedal"], 1.73, no_events, no_events)
        self.assertEqual(result, [{"start": 0., "end": .6}, {"start": 1.2, "end": 1.73}])
        output["pedal"][:] = 0
        output["pedal"][10:12] = .9
        self.assertEqual(decode_pedal_events(output["pedal"], 2., no_events, no_events), [])
        self.assertEqual(decode_pedal_events(np.zeros((0, 1)), 0, np.zeros((0, 1))), [])

    def test_legacy_five_heads_keep_existing_pedal_behavior_and_note_events(self):
        output = self.probabilities(50)
        output["pedal"][10:45] = .9
        output["frame"][5:30, 39] = .9
        output["onset"][5, 39] = .9
        output["offset"][30, 39] = .9
        old = decode_outputs(output, 1., .5)
        self.assertEqual(old["pedals"], [{"start": .2, "end": .9}])
        output["pedal_onset"] = np.zeros((50, 1), np.float32)
        output["pedal_offset"] = np.zeros((50, 1), np.float32)
        new = decode_outputs(output, 1., .5)
        self.assertEqual(old["notes"], new["notes"])
        self.assertEqual(old["pedals"], new["pedals"])
        self.assertNotIn("pedals", decode_outputs(output, 1., .5, include_details=False))


if __name__ == "__main__":
    unittest.main()
