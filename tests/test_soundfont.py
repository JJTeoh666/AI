import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from piano_ml.soundfont import PIANO_RELEASE_SECONDS, SampledPiano, _events, sampled_piano_available
from piano_ml.synthesis import render_result_wav


class FakePiano:
    instances = []

    def __init__(self, rate):
        self.rate, self.position = rate, 0
        self.events, self.closed = [], False
        self.instances.append(self)

    def event(self, kind, key, value):
        self.events.append((self.position, kind, key, value))

    def render(self, frames):
        self.position += frames
        return np.zeros((frames, 2), np.float32)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True


class SoundfontTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_unmatched_note_release_is_harmless_but_other_failures_raise(self):
        piano = SampledPiano.__new__(SampledPiano)
        piano.synth, piano.lib = 1, Mock()
        piano.lib.fluid_synth_noteoff.return_value = -1
        piano.event("off", 95, 0)
        piano.lib.fluid_synth_noteoff.assert_called_once_with(1, 0, 95)
        for kind, function, key, value in (("on", "fluid_synth_noteon", 95, 80),
                                            ("cc", "fluid_synth_cc", 64, 127)):
            getattr(piano.lib, function).return_value = -1
            with self.subTest(kind=kind), self.assertRaises(OSError):
                piano.event(kind, key, value)
        with self.assertRaises(ValueError):
            piano.event("off", 128, 0)
        with self.assertRaises(ValueError):
            piano.event("unknown", 95, 0)
        piano.synth = None
        with self.assertRaises(OSError):
            piano.event("off", 95, 0)

    def test_same_pitch_duplicates_and_rounded_collisions_merge_without_early_release(self):
        for second_start in (.1, .100001):
            result = {"duration": 1, "notes": [
                {"pitch": 95, "start": .1, "end": .5, "velocity": 80},
                {"pitch": 95, "start": second_start, "end": .8, "velocity": 50}]}
            events = [event for event in _events(result, 44100) if event[2] != "cc"]
            self.assertEqual(events, [(4410, 3, "on", 95, 80), (35280, 2, "off", 95, 0)])

    def test_adjacent_sample_repeated_strikes_release_before_next_onset(self):
        result = {"duration": 1, "notes": [
            {"pitch": 95, "start": .1, "end": .9, "velocity": 80},
            {"pitch": 95, "start": 4411 / 44100, "end": .5, "velocity": 60}]}
        events = [event for event in _events(result, 44100) if event[2] != "cc"]
        self.assertEqual(events, [(4410, 3, "on", 95, 80), (4411, 2, "off", 95, 0),
                                  (4411, 3, "on", 95, 60), (22050, 2, "off", 95, 0)])

    def test_stereo_events_repeated_keys_velocities_and_overlapping_pedal(self):
        result = {"duration": 1.4, "notes": [
            {"pitch": 60, "start": 0.2, "end": 0.8, "velocity": 40},
            {"pitch": 60, "start": 0.5, "end": 1.0, "velocity": 110}],
            "pedals": [{"start": 0.25, "end": 0.9}, {"start": 0.8, "end": 1.1}]}
        output = self.root / "sampled.wav"
        progress = []
        with patch("piano_ml.soundfont.SampledPiano", FakePiano):
            render_result_wav(result, output, engine="sampled",
                              progress=lambda done, total: progress.append((done, total)))
        piano = FakePiano.instances[-1]
        self.assertTrue(piano.closed)
        events = [(frame / piano.rate, kind, key, value) for frame, kind, key, value in piano.events]
        self.assertIn((0.2, "on", 60, 40), events)
        self.assertLess(events.index((0.5, "off", 60, 0)), events.index((0.5, "on", 60, 110)))
        self.assertEqual([(time, value) for time, kind, key, value in events if kind == "cc"],
                         [(0.25, 127), (1.1, 0), (1.4, 0)])
        with wave.open(str(output), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (2, 2, 44100))
            self.assertEqual(wav.getnframes(), round((1.4 + PIANO_RELEASE_SECONDS) * 44100))
        self.assertEqual(progress[-1], (4, 4))
        self.assertEqual(result["notes"][0]["end"], 0.8)

    def test_cancellation_closes_synth_and_preserves_previous_export(self):
        output = self.root / "existing.wav"
        output.write_bytes(b"previous export")
        cancel = threading.Event()
        result = {"duration": 2, "notes": [{"pitch": 60, "start": 0, "end": 1.5}]}
        with patch("piano_ml.soundfont.SampledPiano", FakePiano), self.assertRaises(InterruptedError):
            render_result_wav(result, output, engine="sampled", cancel=cancel,
                              progress=lambda *_: cancel.set())
        self.assertTrue(FakePiano.instances[-1].closed)
        self.assertEqual(output.read_bytes(), b"previous export")
        self.assertFalse(output.with_name("existing.wav.partial").exists())

    def test_auto_engine_uses_installed_piano(self):
        output = self.root / "auto.wav"
        result = {"duration": 1, "notes": [{"pitch": 60, "start": 0.1, "end": 0.5}]}
        with patch("piano_ml.synthesis.sampled_piano_available", return_value=True), \
                patch("piano_ml.synthesis.render_sampled_wav", return_value=output) as renderer:
            self.assertEqual(render_result_wav(result, output), output)
        self.assertEqual(renderer.call_args.args[2], 44100)

    @unittest.skipUnless(sampled_piano_available(), "Download piano assets to run the real sample integration test")
    def test_real_long_b6_release_renders_complete_non_silent_wav(self):
        result = {"duration": 9, "notes": [{"pitch": 95, "start": 0, "end": 8, "velocity": 64}]}
        output = render_result_wav(result, self.root / "held-b6.wav", engine="sampled")
        with wave.open(str(output), "rb") as audio:
            self.assertEqual(audio.getnframes(), round((9 + PIANO_RELEASE_SECONDS) * 44100))
            self.assertGreater(np.max(np.abs(np.frombuffer(audio.readframes(44100), dtype="<i2").astype(float))), 0)
        self.assertFalse(output.with_name(output.name + ".partial").exists())

    @unittest.skipUnless(sampled_piano_available(), "Download piano assets to run the real sample integration test")
    def test_real_piano_stereo_pitch_velocity_pedal_and_loud_chord(self):
        result = {"duration": 1.2, "notes": [{"pitch": 69, "start": 0.15, "end": 0.35, "velocity": 40}],
                  "pedals": [{"start": 0.2, "end": 0.9}]}
        def audio(name):
            path = render_result_wav(result, self.root / name)
            with wave.open(str(path), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getframerate()), (2, 44100))
                return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, 2).astype(float) / 32768
        quiet = audio("quiet.wav")
        result["notes"][0]["velocity"] = 110
        loud = audio("loud.wav")
        interval = slice(round(0.18 * 44100), round(0.3 * 44100))
        self.assertGreater(np.sqrt(np.mean(loud[interval] ** 2)), np.sqrt(np.mean(quiet[interval] ** 2)) * 1.5)
        self.assertEqual(np.max(abs(loud[:round(0.14 * 44100)])), 0)
        self.assertGreater(np.sqrt(np.mean((loud[:, 0] - loud[:, 1]) ** 2)), 0.0001)
        segment = loud[round(0.18 * 44100):round(0.32 * 44100)].mean(axis=1)
        frequencies = np.fft.rfftfreq(len(segment), 1 / 44100)
        spectrum = abs(np.fft.rfft(segment * np.hanning(len(segment))))
        self.assertAlmostEqual(frequencies[np.argmax(spectrum)], 440, delta=10)
        result["pedals"] = []
        dry = audio("no-pedal.wav")
        after_release = slice(round(0.6 * 44100), round(0.8 * 44100))
        self.assertGreater(np.sqrt(np.mean(loud[after_release] ** 2)), np.sqrt(np.mean(dry[after_release] ** 2)) * 1.5)
        with SampledPiano(44100) as piano:
            for pitch in (36, 48, 60, 64, 67, 72, 76, 79):
                piano.event("on", pitch, 127)
            chord = piano.render(44100)
        self.assertTrue(np.isfinite(chord).all())
        self.assertLess(float(np.max(abs(chord))), 0.99)


if __name__ == "__main__":
    unittest.main()
