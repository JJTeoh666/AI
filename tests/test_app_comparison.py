import json
import tempfile
import tkinter as tk
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from piano_ml.app import PianoApp


class FakePlayer:
    def __init__(self):
        self.calls = []
        self.is_playing = False

    def play(self, path, start_seconds=0):
        self.calls.append((path, start_seconds))
        self.is_playing = True
        return 8

    def stop(self):
        self.is_playing = False

    def set_volume(self, value):
        self.volume = value

    def set_gain(self, value):
        self.gain = value


class AppComparisonTest(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        with patch.object(PianoApp, "refresh_models"), patch("piano_ml.app.playback_available", return_value=False):
            self.app = PianoApp(self.root)
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.directory = Path(self.temp.name)
        self.audio = self.directory / "recording.wav"
        with wave.open(str(self.audio), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(8 * 16000 * 2))
        self.app.audio.set(str(self.audio))

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def result(self, basic=False):
        return {"audio": str(self.audio), "duration": 8, "model": "Spotify Basic Pitch" if basic else "test-v8.1.pt",
                "engine": "basic-pitch" if basic else "piano", "analysis_seconds": .4,
                "thresholds": {"frame": .3, "onset": .5} if basic else {"frame": .67, "onset": .41, "offset": .72},
                "notes": [{"name": "C4", "pitch": 60, "start": 0, "end": 1, "confidence": .8}],
                "chords": [], "pedals": [], "inference_config": {"overlap_seconds": 3, "overlap_blend": "equal"}}

    def load_both(self):
        self.app.show_result(self.result())
        self.app.show_result(self.result(True))

    def test_basic_result_keeps_own_settings_and_both_results(self):
        self.app.show_result(self.result())
        settings = (self.app.threshold.get(), self.app.onset_threshold.get(), self.app.offset_threshold.get(),
                    self.app.overlap_seconds.get(), self.app.overlap_method.get())
        self.app.show_result(self.result(True))
        self.assertEqual(settings, (self.app.threshold.get(), self.app.onset_threshold.get(), self.app.offset_threshold.get(),
                                    self.app.overlap_seconds.get(), self.app.overlap_method.get()))
        self.assertEqual(set(self.app.results), {"own", "basic"})
        self.assertEqual(self.app.selected_result.get(), "Basic Pitch")
        self.assertIn("No pedal detection", self.app.summary.get())
        self.assertEqual(self.app.volume.get(), 50)

    def test_pause_seek_and_switch_keep_position_with_distinct_audio_caches(self):
        self.load_both()
        player = self.app.player = FakePlayer()
        self.app.audio_cache = {"own": self.audio, "basic": self.directory / "basic.wav"}
        self.app.audio_cache["basic"].touch()
        with patch("piano_ml.app.time.monotonic", return_value=100):
            self.app.selected_result.set("Your model")
            self.app.select_result()
            self.app.play_selected()
        with patch("piano_ml.app.time.monotonic", return_value=102):
            self.app.playback_source.set("Basic Pitch")
            self.app.change_source()
            self.assertEqual(player.calls[-1], (self.app.audio_cache["basic"], 2))
            self.assertEqual(self.app.play_result_button["text"], "Pause")
        with patch("piano_ml.app.time.monotonic", return_value=103):
            self.app.toggle_play()
            self.assertEqual(self.app.playback_position, 3)
            self.assertFalse(player.is_playing)
            self.assertEqual(self.app.play_result_button["text"], "Play")
            self.app.seek_to(5)
            self.app.toggle_play()
            self.assertEqual(player.calls[-1][1], 5)
            self.app.playback_source.set("Original recording")
            self.app.change_source()
            self.assertEqual(player.calls[-1], (self.audio, 5))
        self.app.stop()
        self.assertEqual(self.app.seek_value.get(), 0)

    def test_new_recording_and_import_clear_comparison(self):
        self.load_both()
        with patch("piano_ml.app.filedialog.askopenfilename", return_value=str(self.audio)):
            self.app.browse_audio()
        self.assertEqual(self.app.results, {})
        self.assertIsNone(self.app.result)
        self.assertEqual(self.app.audio_cache, {})
        self.assertEqual(self.app.playing_duration, 8)
        self.load_both()
        path = self.directory / "basic.json"
        path.write_text(json.dumps(self.result(True)), encoding="utf-8")
        with patch("piano_ml.app.filedialog.askopenfilename", return_value=str(path)):
            self.app.open_json()
        self.assertEqual(set(self.app.results), {"basic"})

    def test_basic_needs_no_pytorch_checkpoint_and_busy_disables_inputs(self):
        with patch("piano_ml.app.threading.Thread") as thread:
            self.app.analyze_basic()
        self.assertEqual(thread.call_args.kwargs["args"], (self.audio,))
        for widget in (self.app.compare_button, self.app.model_browse_button, self.app.audio_browse_button,
                       self.app.source_box, self.app.models_box, self.app.seek_slider):
            self.assertEqual(str(widget["state"]), "disabled")
        with patch("piano_ml.app.transcribe_basic_pitch", return_value=self.result(True)) as transcribe, \
                patch("piano_ml.app.waveform_envelope", return_value=None):
            self.app._basic_worker(self.audio)
        self.assertIs(transcribe.call_args.kwargs["cancel"], self.app.cancel_event)
        self.assertEqual(self.app.messages.get_nowait()[0], "result")

    def test_collapsible_settings_and_fixed_player_fit_minimum_window(self):
        self.root.attributes("-alpha", 0)
        self.root.deiconify()
        for width, height in ((900, 650), (1260, 850)):
            self.root.geometry(f"{width}x{height}")
            self.root.update()
            self.assertEqual(self.app.advanced.winfo_manager(), "")
            self.app.toggle_settings()
            self.root.update_idletasks()
            self.assertEqual(self.app.advanced.winfo_manager(), "pack")
            self.app.toggle_settings()
            self.root.update_idletasks()
            for widget in (self.app.source_box, self.app.play_result_button, self.app.seek_slider,
                           self.app.export_audio_button, self.app.gain_slider, self.app.volume_slider,
                           self.app.analyze_button, self.app.compare_button, self.app.cancel_button,
                           self.app.model_browse_button, self.app.audio_browse_button):
                right = widget.winfo_rootx() + widget.winfo_width() - self.root.winfo_rootx()
                self.assertLessEqual(right, width, str(widget))
                self.assertGreaterEqual(widget.winfo_width(), widget.winfo_reqwidth(), str(widget))
            self.assertEqual(self.app.playback_controls.winfo_manager(), "grid")
            self.assertLess(self.app.playback_controls.winfo_y() + self.app.playback_controls.winfo_height(), height)

    def test_dragging_while_playing_restarts_once_on_release(self):
        self.load_both()
        self.app.player = FakePlayer()
        self.app.audio_cache["basic"] = self.audio
        with patch("piano_ml.app.time.monotonic", return_value=100):
            self.app.play_selected()
            self.app.begin_seek()
            self.app.seek_value.set(4)
            self.app.preview_seek(4)
            self.assertEqual(len(self.app.player.calls), 1)
            self.app.end_seek()
            self.assertEqual(len(self.app.player.calls), 2)
            self.assertEqual(self.app.player.calls[-1][1], 4)

    def test_cancelled_ready_message_does_not_play_or_replace_result(self):
        self.app.audio_cancel.set()
        self.app.messages.put(("audio_ready", (self.audio, None, "basic")))
        with patch.object(self.app, "start_playback") as play:
            self.app.poll()
        play.assert_not_called()
        self.app.cancel_event.set()
        self.app.messages.put(("result", (self.result(True), None)))
        self.app.poll()
        self.assertEqual(self.app.results, {})


if __name__ == "__main__":
    unittest.main()
