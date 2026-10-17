import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.bytedance import (MODEL_PATH, OUTPUT_WIDTHS, chord_spans, decode_probabilities,
                                predict_probabilities, read_audio, resolve_device, segment_plan,
                                transcribe_bytedance, verify_checkpoint)
from piano_ml.viewer import read_prediction


def blank_outputs(frames=100):
    return {key: np.zeros((frames, width), np.float32) for key, width in OUTPUT_WIDTHS.items()}


class TimelineModel:
    def __call__(self, wave):
        stamps = wave[0, 0] + torch.arange(1001) * 160
        return {key: stamps[None, :, None].expand(1, 1001, width) + index
                for index, (key, width) in enumerate(OUTPUT_WIDTHS.items())}


class ByteDanceTest(unittest.TestCase):
    def test_official_plan_and_every_head_alignment_short_partial_exact_and_multiple(self):
        self.assertEqual(list(segment_plan(320000)), [(0, 0, 750), (80000, 250, 750), (160000, 250, 1000)])
        for length in (1, 159, 160, 159999, 160000, 160001, 320000, 341113):
            with self.subTest(length=length):
                progress = []
                output = predict_probabilities(np.arange(length, dtype=np.float32), TimelineModel(), "cpu",
                                               progress=lambda done, total: progress.append((done, total)))
                expected = min(length // 160 + 1, 1000 * max(1, (length + 159999) // 160000))
                if length == 160000:
                    expected = 1001
                for index, (key, width) in enumerate(OUTPUT_WIDTHS.items()):
                    self.assertEqual(output[key].shape, (expected, width))
                    np.testing.assert_array_equal(output[key][:, 0], np.arange(expected) * 160 + index)
                self.assertEqual(progress[-1], (len(list(segment_plan(length))),) * 2)

    def test_repeated_strikes_velocity_offset_and_pedal_decoder(self):
        output = blank_outputs()
        for pitch in (39, 43, 46):
            output["frame_output"][5:65, pitch] = .8
            output["reg_onset_output"][[5, 35], pitch] = .9
            output["reg_offset_output"][55, pitch] = .9
            output["velocity_output"][:, pitch] = .6
        output["pedal_frame_output"][10:80] = .8
        output["reg_pedal_offset_output"][75] = .9
        notes, pedals = decode_probabilities(output, 1.)
        self.assertEqual(len(notes), 6)
        self.assertEqual([n["pitch"] for n in notes], [60, 64, 67] * 2)
        np.testing.assert_allclose([n["end"] for n in notes], [.34] * 3 + [.55] * 3, atol=1e-6)
        self.assertTrue(all(n["velocity"] == 76 and abs(n["confidence"] - .8) < 1e-6 for n in notes))
        np.testing.assert_allclose([[p["start"], p["end"]] for p in pedals], [[.1, .75]], atol=1e-6)

    def test_silence_short_and_recording_boundary_closes_notes_and_pedal(self):
        for length in (1, 3, 100):
            self.assertEqual(decode_probabilities(blank_outputs(length), .001 if length < 4 else 1), ([], []))
        output = blank_outputs(101)
        output["reg_onset_output"][5, 39] = .9
        output["frame_output"][5:, 39] = .8
        output["velocity_output"][:, 39] = .7
        output["pedal_frame_output"][10:] = .9
        notes, pedals = decode_probabilities(output, 1)
        self.assertEqual(notes[0]["end"], 1)
        self.assertEqual(pedals[0]["end"], 1)

    def test_cancel_before_loading_between_windows_and_in_decoder(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            transcribe_bytedance("does-not-exist.wav", cancel=cancel)
        with self.assertRaises(InterruptedError):
            decode_probabilities(blank_outputs(), 1, cancel)
        cancel.clear()
        with self.assertRaises(InterruptedError):
            predict_probabilities(np.zeros(320000, np.float32), TimelineModel(), "cpu", cancel=cancel,
                                  progress=lambda *_: cancel.set())

    def test_audio_resampling_amplitude_and_checkpoint_errors(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "short.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(np.full((8, 2), 16384, dtype="<i2").tobytes())
            samples, duration = read_audio(path)
            self.assertEqual(duration, .001)
            self.assertEqual(len(samples), 16)
            self.assertLess(float(samples.max()), .6)
            with patch("piano_ml.bytedance._load_model", return_value=lambda _: {
                    key: torch.zeros(1, 1001, width) for key, width in OUTPUT_WIDTHS.items()}):
                result = transcribe_bytedance(path, device="cpu")
            self.assertEqual(result["notes"], [])
            self.assertTrue(result["pedal_detection"])
            with self.assertRaises(ValueError):
                verify_checkpoint(path)
            with self.assertRaises(FileNotFoundError):
                verify_checkpoint(Path(directory) / "missing.pth")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaises(ValueError):
                resolve_device("cuda")

    def test_chords_count_overlapping_same_pitch_voices(self):
        notes = [{"pitch": pitch, "start": 0, "end": 1} for pitch in (60, 64, 67)]
        notes.append({"pitch": 60, "start": .5, "end": 2})
        self.assertEqual(chord_spans(notes), [{"name": "C major", "start": 0, "end": 1}])

    @unittest.skipUnless(MODEL_PATH.is_file() and Path("diagnostics/browser-clip.wav").is_file(), "Downloaded model and clip")
    def test_real_model_result_roundtrip(self):
        result = transcribe_bytedance("diagnostics/browser-clip.wav", device="cpu")
        self.assertEqual(result["duration"], 8)
        self.assertGreater(len(result["notes"]), 10)
        self.assertTrue(result["pedal_detection"])
        for event in result["notes"] + result["pedals"]:
            self.assertTrue(0 <= event["start"] < event["end"] <= 8)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(read_prediction(path)["engine"], "bytedance")


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
