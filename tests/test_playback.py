import ctypes
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from piano_ml.playback import WavPlayer, playback_available, scale_pcm16


class ControlledOutput:
    """Keep the first two buffers queued until the test changes the volume."""

    def __init__(self, rate, channels):
        self.blocks = []
        self.queued = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def write(self, data):
        self.blocks.append(data)
        if len(self.blocks) == 2:
            self.queued.set()

    def collect(self):
        return 2 if self.queued.is_set() and not self.release.is_set() else 0

    def close(self):
        self.closed = True


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.path = Path(self.temp.name) / "stereo.wav"
        samples = np.tile(np.array([12000, -8000], dtype="<i2"), 8000)
        with wave.open(str(self.path), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(samples.tobytes())

    def tearDown(self):
        self.temp.cleanup()

    def test_stereo_gain_mute_and_full_volume_preserve_pcm_range(self):
        samples = np.array([-32768, 32767, -20000, 10000, 0, 2], dtype="<i2")
        data = samples.tobytes()
        self.assertEqual(scale_pcm16(data, 100), data)
        self.assertEqual(scale_pcm16(data, 0), bytes(len(data)))
        np.testing.assert_array_equal(np.frombuffer(scale_pcm16(data, 50), dtype="<i2"),
                                      [-16384, 16384, -10000, 5000, 0, 1])

    def test_gain_amplifies_quiet_audio_and_limits_peaks_without_wrapping(self):
        samples = np.array([1000, -2000, 10000, -10000, 0, 32000], dtype="<i2")
        boosted = np.frombuffer(scale_pcm16(samples.tobytes(), 100, 20), dtype="<i2")
        np.testing.assert_array_equal(boosted, [10000, -20000, 32767, -32768, 0, 32767])
        np.testing.assert_array_equal(
            np.frombuffer(scale_pcm16(samples[:2].tobytes(), 50, 20), dtype="<i2"), [5000, -10000])
        self.assertEqual(scale_pcm16(samples.tobytes(), 0, 24), bytes(len(samples.tobytes())))

    def test_gain_can_change_during_playback_without_modifying_the_wav(self):
        original = self.path.read_bytes()
        output = ControlledOutput(16000, 2)
        player = WavPlayer()  # Starts at 50% volume, with zero gain.
        with patch("piano_ml.playback._WaveOutput", return_value=output):
            player.play(self.path)
            try:
                self.assertTrue(output.queued.wait(1))
                for block in output.blocks:
                    np.testing.assert_array_equal(np.frombuffer(block, dtype="<i2").reshape(-1, 2)[0],
                                                  [6000, -4000])
                player.set_gain(6)
                output.release.set()
                player._thread.join(2)
                self.assertFalse(player.is_playing)
                for block in output.blocks[2:]:
                    np.testing.assert_array_equal(np.frombuffer(block, dtype="<i2").reshape(-1, 2)[0],
                                                  [11972, -7981])
                self.assertEqual(sum(map(len, output.blocks)), 32000)
            finally:
                player.stop()
        self.assertEqual(self.path.read_bytes(), original)

    def test_volume_changes_during_streaming_and_source_is_unchanged(self):
        original = self.path.read_bytes()
        output = ControlledOutput(16000, 2)
        player = WavPlayer()
        player.set_volume(0)
        with patch("piano_ml.playback._WaveOutput", return_value=output):
            self.assertEqual(player.play(self.path), 0.5)
            try:
                self.assertTrue(output.queued.wait(1))
                self.assertTrue(player.is_playing)
                self.assertEqual(output.blocks, [bytes(3200), bytes(3200)])
                player.set_volume(50)
                output.release.set()
                player._thread.join(2)
                self.assertFalse(player._thread.is_alive())
                self.assertFalse(player.is_playing)
                for block in output.blocks[2:]:
                    np.testing.assert_array_equal(np.frombuffer(block, dtype="<i2").reshape(-1, 2)[0],
                                                  [6000, -4000])
                self.assertEqual(sum(map(len, output.blocks)), 32000)
                self.assertTrue(output.closed)
            finally:
                player.stop()
        self.assertEqual(self.path.read_bytes(), original)

    def test_stop_releases_queued_playback_and_the_file(self):
        output = ControlledOutput(16000, 2)
        player = WavPlayer()
        with patch("piano_ml.playback._WaveOutput", return_value=output):
            player.play(self.path)
            try:
                self.assertTrue(output.queued.wait(1))
            finally:
                player.stop()
        self.assertTrue(output.closed)
        self.assertIsNone(player._thread)
        self.assertFalse(player.is_playing)
        # Windows refuses deletion if the worker still holds the WAV open.
        self.path.unlink()

    def test_device_failure_is_reported_and_closes_stream(self):
        output = ControlledOutput(16000, 2)
        output.write = lambda data: (_ for _ in ()).throw(OSError("device failure"))
        errors = []
        notified = threading.Event()
        def on_error(error):
            errors.append(error)
            notified.set()
        player = WavPlayer(on_error=on_error)
        with patch("piano_ml.playback._WaveOutput", return_value=output):
            player.play(self.path)
            self.assertTrue(notified.wait(1))
            player.stop()
        self.assertEqual(errors, ["device failure"])
        self.assertTrue(output.closed)
        self.path.unlink()

    @unittest.skipUnless(playback_available(), "Windows audio backend")
    def test_real_windows_output_finishes_muted_playback(self):
        if ctypes.WinDLL("winmm").waveOutGetNumDevs() == 0:
            self.skipTest("No Windows audio output devices")
        errors = []
        player = WavPlayer(on_error=errors.append)
        player.set_volume(0)
        player.play(self.path)
        try:
            player._thread.join(3)
            self.assertFalse(player._thread.is_alive())
        finally:
            player.stop()
        self.assertEqual(errors, [])
        # Exercise reset/unprepare while the real driver still owns buffers.
        player.play(self.path)
        try:
            threading.Event().wait(0.07)
            self.assertTrue(player.is_playing)
        finally:
            player.stop()
        self.assertFalse(player.is_playing)
        self.assertEqual(errors, [])
        self.path.unlink()


if __name__ == "__main__":
    unittest.main()
