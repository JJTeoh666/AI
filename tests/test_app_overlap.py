import json
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from piano_ml.app import PianoApp
from piano_ml.viewer import read_prediction
from piano_ml.thresholds import Thresholds


class AppOverlapTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk unavailable: {error}")
        self.root.withdraw()
        with patch.object(PianoApp, "refresh_models"), patch("piano_ml.app.playback_available", return_value=False):
            self.app = PianoApp(self.root)
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.directory = Path(self.temp.name)
        self.audio, self.model = self.directory / "audio.wav", self.directory / "model.pt"
        self.audio.touch()
        self.model.touch()
        self.app.audio.set(str(self.audio))
        self.app.model_paths["test"] = self.model
        self.app.model.set("test")

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    @staticmethod
    def result():
        return {"duration": 8.04, "threshold": 0.5, "notes": [], "chords": [], "pedals": [],
                "inference_config": {"chunk_seconds": 4, "context_seconds": 2, "overlap_seconds": 4,
                                     "overlap_blend": "weighted"}}

    def test_defaults_and_busy_controls(self):
        self.assertEqual(self.app.overlap_seconds.get(), 2)
        self.assertEqual(self.app.overlap_method.get(), "Weighted average")
        self.assertEqual(tuple(self.app.overlap_method_box["values"]), ("Weighted average", "50/50 average", "Keep center"))
        for busy, state in ((True, "disabled"), (False, "readonly")):
            self.app.set_busy(busy, "analysis")
            self.assertEqual(str(self.app.overlap_method_box["state"]), state)
            self.assertEqual(str(self.app.overlap_box["state"]), state)

    def test_v81_model_is_preferred_and_version_is_displayed(self):
        checkpoints = self.directory / "checkpoints"
        checkpoints.mkdir()
        for name in ("piano-v8.pt", "piano-v8.1.pt"):
            (checkpoints / name).touch()
        self.app.model.set("")
        self.app.model_paths.clear()
        with patch("piano_ml.app.PROJECT", self.directory), patch("piano_ml.app.load_model") as load:
            load.return_value = SimpleNamespace(model_version="8.1", decoding_thresholds=Thresholds())
            self.app.refresh_models()
            self.assertEqual(self.app.model.get(), "piano-v8.1.pt")
            self.assertTrue(self.app.threshold_details.get().startswith("V8.1"))
            (checkpoints / "piano-v8.1.pt").unlink()
            self.app.model.set("")
            self.app.model_paths.clear()
            load.return_value.model_version = "8"
            self.app.refresh_models()
            self.assertEqual(self.app.model.get(), "piano-v8.pt")
            self.assertTrue(self.app.threshold_details.get().startswith("V8 —"))

    def test_selection_reaches_the_inference_worker(self):
        for label, blend in (("Weighted average", "weighted"), ("50/50 average", "equal"), ("Keep center", "crop")):
            with self.subTest(blend=blend):
                self.app.overlap_seconds.set(4)
                self.app.overlap_method.set(label)
                with patch("piano_ml.app.threading.Thread") as thread:
                    self.app.analyze()
                    worker_args = thread.call_args.kwargs["args"]
                    self.assertEqual(worker_args[-2:], (4, blend))
                    thread.return_value.start.assert_called_once()
                with patch("piano_ml.app.transcribe_audio", return_value=self.result()) as transcribe, \
                        patch("piano_ml.app.waveform_envelope", return_value=None):
                    self.app._worker(*worker_args)
                self.assertEqual(transcribe.call_args.kwargs["context_seconds"], 2)
                self.assertEqual(transcribe.call_args.kwargs["overlap_blend"], blend)
                self.assertEqual(self.app.messages.get_nowait()[0], "result")
                self.app.set_busy(False)

    def test_saved_and_legacy_results_restore_the_correct_method(self):
        path = self.directory / "prediction.json"
        for blend, label in (("weighted", "Weighted average"), ("equal", "50/50 average"), ("crop", "Keep center")):
            result = self.result()
            result["inference_config"]["overlap_blend"] = blend
            path.write_text(json.dumps(result), encoding="utf-8")
            self.app.overlap_method.set("Keep center")
            self.app.show_result(read_prediction(path))
            self.assertEqual(self.app.overlap_method.get(), label)
            self.assertEqual(self.app.overlap_seconds.get(), 4)
        del result["inference_config"]["overlap_blend"]
        self.app.show_result(result)
        self.assertEqual(self.app.overlap_method.get(), "Keep center")
        self.app.overlap_method.set("Weighted average")
        del result["inference_config"]
        self.app.show_result(result)
        self.assertEqual(self.app.overlap_method.get(), "Keep center")


if __name__ == "__main__":
    unittest.main()
