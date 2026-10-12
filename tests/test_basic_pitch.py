import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from piano_ml.basic_pitch import (MODEL_PATH, HOP_SAMPLES, PADDING, audio_window,
                                  chord_spans, decode_notes, frame_time, read_audio, transcribe_basic_pitch)
from piano_ml.viewer import read_prediction


class BasicPitchTest(unittest.TestCase):
    def test_windows_are_aligned_and_partial_windows_zero_padded(self):
        samples = np.arange(90000, dtype=np.float32)
        first = audio_window(samples, 0).ravel()
        np.testing.assert_array_equal(first[:PADDING], 0)
        np.testing.assert_array_equal(first[PADDING:], samples[:len(first) - PADDING])
        second = audio_window(samples, 1).ravel()
        np.testing.assert_array_equal(second, samples[HOP_SAMPLES - PADDING:HOP_SAMPLES - PADDING + len(second)])
        last = audio_window(samples, 3).ravel()
        np.testing.assert_array_equal(last, 0)

    def test_silence_short_audio_and_cancel(self):
        self.assertEqual(decode_notes(np.zeros((80, 88)), np.zeros((80, 88))), [])
        self.assertEqual(decode_notes(np.zeros((0, 88)), np.zeros((0, 88))), [])
        self.assertEqual(decode_notes(np.ones((1, 88)), np.ones((1, 88))), [])
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            transcribe_basic_pitch("does-not-exist.wav", cancel=cancel)

    def test_polyphony_and_repeated_strikes_follow_official_decoder(self):
        frames, onsets = np.zeros((90, 88)), np.zeros((90, 88))
        for pitch in (39, 43, 46):
            frames[5:65, pitch] = .8
            onsets[[5, 35], pitch] = .9
        events = sorted(decode_notes(frames, onsets))
        self.assertEqual([(s, e, p) for s, e, p, a in events],
                         [(5, 35, 60), (5, 35, 64), (5, 35, 67),
                          (35, 65, 60), (35, 65, 64), (35, 65, 67)])
        self.assertTrue(all(abs(event[3] - .8) < 1e-6 for event in events))

    def test_melodia_recovers_notes_without_onset_peaks(self):
        frames, onsets = np.zeros((90, 88)), np.zeros((90, 88))
        frames[5:60, 39] = .8
        events = decode_notes(frames, onsets)
        self.assertEqual([(s, e, p) for s, e, p, _ in events], [(5, 59, 60)])
        self.assertGreater(frame_time(172), frame_time(171))

    def test_chords_count_overlapping_same_pitch_voices(self):
        notes = [{"pitch": pitch, "start": 0, "end": 1} for pitch in (60, 64, 67)]
        notes.append({"pitch": 60, "start": .5, "end": 2})
        self.assertEqual(chord_spans(notes), [{"name": "C major", "start": 0, "end": 1}])

    def test_resampling_preserves_amplitude_and_handles_short_recording(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "short.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(np.full((8, 2), 16384, dtype="<i2").tobytes())
            samples, duration = read_audio(path)
            self.assertAlmostEqual(duration, .0005)
            self.assertEqual(len(samples), 12)
            self.assertLessEqual(float(samples.max()), .6)
            with patch("piano_ml.basic_pitch._session") as session:
                session.return_value.run.return_value = [np.zeros((1, 172, 88)), np.zeros((1, 172, 88))]
                result = transcribe_basic_pitch(path)
            self.assertEqual(result["notes"], [])
            self.assertFalse(result["pedal_detection"])

    @unittest.skipUnless(MODEL_PATH.is_file() and Path("diagnostics/browser-clip.wav").is_file(), "Downloaded model and clip")
    def test_real_model_clip_result_roundtrip_and_cancel_between_windows(self):
        progress = []
        result = transcribe_basic_pitch("diagnostics/browser-clip.wav", progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(result["duration"], 8)
        self.assertGreater(len(result["notes"]), 10)
        self.assertEqual(progress[-1][0], progress[-1][1])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(read_prediction(path)["engine"], "basic-pitch")
        cancel = threading.Event()
        with self.assertRaises(InterruptedError):
            transcribe_basic_pitch("diagnostics/browser-clip.wav", cancel=cancel,
                                   progress=lambda done, total: cancel.set())


if __name__ == "__main__":
    unittest.main()
