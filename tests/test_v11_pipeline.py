import csv
import io
import json
import os
import threading
import unittest
import wave
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.nn import functional as F

from piano_ml.__main__ import main
from piano_ml.data import MaestroWindows
from piano_ml.evaluation import evaluate
from piano_ml.inference import load_model, predict_probabilities, transcribe_audio
from piano_ml.model import (STFTCQTPianoNet, STFTFANSustainPianoNet, V11_ARCHITECTURE,
                            build_model)
from piano_ml.training import (LOSS_COMPONENTS, train, training_loss,
                               training_loss_components)
from tests import test_v8 as v8, test_v9 as v9


def _vlq(value):
    encoded = [value & 0x7f]
    while value >> 7:
        value >>= 7
        encoded.insert(0, 0x80 | (value & 0x7f))
    return bytes(encoded)


class V11PipelineTest(unittest.TestCase):
    tearDown = v8.V8PipelineTest.tearDown
    test_selection_scheduling_and_requested_epochs = v9.V9PipelineTest.test_avgf_selection_scheduler_and_no_early_stop
    test_resume_restores_optimizer_scheduler_and_random_states = v9.V9PipelineTest.test_resume_restores_random_and_optimizer_states_exactly

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        v8.V8PipelineTest.setUp(self)
        # One physical key release at 0.3 s, sustained until pedal-up at 0.8 s.
        for name, pitch in (("bass", 24), ("treble", 96), ("validation", 60), ("test", 96)):
            events, previous = bytearray(), 0
            for tick, message in ((96, bytes((0x90, pitch, 100))),
                                  (115, bytes((0xb0, 64, 127))),
                                  (288, bytes((0x80, pitch, 0))),
                                  (768, bytes((0xb0, 64, 0))),
                                  (960, b"\xff\x2f\x00")):
                events.extend(_vlq(tick - previous))
                events.extend(message)
                previous = tick
            midi = (b"MThd" + (6).to_bytes(4, "big") + bytes((0, 0, 0, 1, 1, 0xe0))
                    + b"MTrk" + len(events).to_bytes(4, "big") + bytes(events))
            (self.root / f"{name}.midi").write_bytes(midi)

    def options(self, output, **changes):
        return v8.V8PipelineTest.options(self, output, **{
            "architecture": V11_ARCHITECTURE, "feature_width": 16, "hidden_size": 8,
            "gru_layers": 1, "fourier_modes": None, "fourier_layers": None, **changes})

    def test_fresh_explicit_v11_versioned_files_defaults_metadata_and_loss_reporting(self):
        previous = Path.cwd()
        console = io.StringIO()
        try:
            os.chdir(self.root)
            args = self.options(None, architecture=V11_ARCHITECTURE, gru_layers=None)
            args.output = None
            with patch("piano_ml.training.MaestroWindows", wraps=MaestroWindows) as datasets, \
                    redirect_stdout(console):
                train(args)
        finally:
            os.chdir(previous)
        best = self.root / "checkpoints/piano-v11.pt"
        latest = self.root / "checkpoints/piano-v11.last.pt"
        history = self.root / "checkpoints/piano-v11.history.jsonl"
        self.assertTrue(best.is_file())
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["format_version"], 13)
        self.assertEqual(saved["model_version"], "11")
        self.assertEqual(saved["model_config"]["architecture"], V11_ARCHITECTURE)
        self.assertEqual(saved["model_config"]["fan_window"], 8192)
        self.assertEqual(saved["model_config"]["short_fft"], 2048)
        self.assertEqual(saved["model_config"]["gru_layers"], 4)
        self.assertEqual(saved["note_duration"], "sustain_extended")
        self.assertEqual(saved["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(saved["training_config"]["threshold_objective"], "pitch_f12345")
        self.assertEqual(saved["training_config"]["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertTrue(saved["training_config"]["pedal_events"])
        self.assertEqual(saved["training_config"]["note_duration"], "sustain_extended")
        self.assertEqual(saved["selection_metric"], "avgf_loss")
        self.assertEqual(saved["selection_mode"], "min")
        self.assertFalse(saved["training_config"]["early_stopping"])
        self.assertEqual(saved["validation"]["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertIn("note_macro_f12345", saved["validation"])
        for call in datasets.call_args_list:
            self.assertEqual(call.kwargs["note_duration"], "sustain_extended")
            self.assertTrue(call.kwargs["pedal_events"])
        record = json.loads(history.read_text())
        self.assertEqual(set(record["train_loss_components"]), set(LOSS_COMPONENTS))
        self.assertAlmostEqual(sum(record["train_loss_components"].values()), record["train_loss"], places=6)
        self.assertEqual(record["train_loss_components"], saved["training_metrics"]["loss_components"])
        self.assertEqual(record["train_frame_bce"], saved["training_metrics"]["frame_bce"])
        self.assertIn("Training V11", console.getvalue())
        self.assertIn("note_macro_f12345=", console.getvalue())
        loaded = load_model(latest, torch.device("cpu"))
        recreated = build_model(saved["model_config"]).eval()
        recreated.load_state_dict(saved["model"])
        with torch.no_grad():
            audio = torch.randn(1, 16001) * 0.1
            first, second = loaded(audio), recreated(audio)
        self.assertEqual(len(first), 7)
        for head in first:
            torch.testing.assert_close(first[head], second[head], atol=0, rtol=0)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu", overlap_blend="equal")
        self.assertEqual(result["model_version"], "11")
        self.assertEqual(result["note_duration"], "sustain_extended")
        self.assertEqual(result["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(result["inference_config"]["overlap_blend"], "equal")

    def test_sustained_targets_and_all_seven_head_losses_contribute(self):
        legacy = MaestroWindows(self.root, "validation", seconds=1, windows_per_file=1,
                               multi_target=True)
        data = MaestroWindows(self.root, "validation", seconds=1, windows_per_file=1,
                             multi_target=True, note_duration="sustain_extended", pedal_events=True)
        audio, labels = data[0]
        _, old_labels = legacy[0]
        self.assertEqual(labels["frame"][20, 39], 1)
        self.assertEqual(old_labels["frame"][20, 39], 0)
        self.assertEqual(labels["offset"][40, 39], 1)
        self.assertEqual(old_labels["offset"][15, 39], 1)
        self.assertAlmostEqual(labels["reference_notes"][0]["end"], 0.8)
        self.assertEqual(set(labels) - {"reference_notes"},
                         {"frame", "onset", "offset", "velocity", "pedal", "pedal_onset", "pedal_offset"})
        self.assertGreater(labels["pedal_onset"].sum(), 0)
        self.assertGreater(labels["pedal_offset"].sum(), 0)
        truth = {head: torch.from_numpy(value)[None] for head, value in labels.items()
                 if head != "reference_notes"}
        model = STFTFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        output = model(torch.from_numpy(audio)[None])
        components = training_loss_components(output, truth, offset_loss_weight=1, release_loss_weight=.25)
        expected_pedal = .2 * F.binary_cross_entropy_with_logits(
            output["pedal"], truth["pedal"], pos_weight=torch.tensor([2.]))
        for head in ("pedal_onset", "pedal_offset"):
            expected_pedal = expected_pedal + .2 * F.binary_cross_entropy_with_logits(
                output[head], truth[head], pos_weight=torch.tensor([10.]))
        torch.testing.assert_close(components["pedal"], expected_pedal)
        objective = training_loss(output, truth, offset_loss_weight=1, release_loss_weight=.25)
        torch.testing.assert_close(objective, sum(components.values()))
        objective.backward()
        for head in (model.pedal_head, model.pedal_onset_head, model.pedal_offset_head):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_cli_rejects_older_initialization_and_supports_compatible_v11(self):
        with patch("sys.argv", ["piano_ml", "train", "--architecture", V11_ARCHITECTURE,
                                "--fan-window", "8192", "--gru-layers", "4"]), \
                patch("piano_ml.__main__.train") as run:
            main()
        self.assertEqual(run.call_args.args[0].fan_window, 8192)
        self.assertEqual(run.call_args.args[0].gru_layers, 4)
        source = self.root / "v101.pt"
        old = STFTCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        torch.save({"model": old.state_dict(), "model_config": old.config}, source)
        with self.assertRaisesRegex(ValueError, "V11 uses long-context FAN"):
            train(self.options(self.root / "bad.pt", init_from=str(source)))
        with self.assertRaisesRegex(ValueError, "replaces the long FFT"):
            train(self.options(self.root / "bad.pt", long_fft=8192))
        fresh = STFTFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        torch.save({"model": fresh.state_dict(), "model_config": fresh.config}, source)
        with redirect_stdout(io.StringIO()):
            train(self.options(self.root / "initialized.pt", init_from=str(source),
                               hidden_size=None, feature_width=None, gru_layers=None))
        saved = torch.load(self.root / "initialized.last.pt", weights_only=True)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["model_config"], fresh.config)

    def test_all_overlap_methods_preserve_seven_heads_and_cancel_between_sections(self):
        path = self.root / "boundary.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(bytes(round(4.13 * 16000) * 2))
        model = STFTFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1).eval()
        zero_overlap = []
        for context in (0, 1, 2):
            for mode in ("crop", "weighted", "equal"):
                with self.subTest(context=context, mode=mode):
                    progress = []
                    duration, outputs = predict_probabilities(
                        path, model, torch.device("cpu"), context_seconds=context, overlap_blend=mode,
                        progress=lambda done, total: progress.append((done, total)))
                    self.assertAlmostEqual(duration, 4.13)
                    self.assertEqual(len(outputs), 7)
                    self.assertEqual(progress, [(1, 2), (2, 2)])
                    for head, values in outputs.items():
                        self.assertEqual(values.shape, (207, 1 if head.startswith("pedal") else 88))
                        self.assertTrue(np.isfinite(values).all())
                        self.assertTrue(((values >= 0) & (values <= 1)).all())
                    if context == 0:
                        zero_overlap.append(outputs)
        for other in zero_overlap[1:]:
            for head in other:
                np.testing.assert_array_equal(other[head], zero_overlap[0][head])
        for preset in (True, False):
            cancel = threading.Event()
            if preset:
                cancel.set()
            with self.assertRaises(InterruptedError):
                predict_probabilities(path, model, torch.device("cpu"), cancel=cancel,
                                      overlap_blend="weighted", progress=lambda *_: cancel.set())

    def test_window_full_evaluation_metadata_csv_and_calibrated_checkpoint(self):
        model = STFTFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        args = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                         threshold=None, thresholds=None, frame_threshold=.55, selection_metric="auto",
                         full_recordings=False, min_note_seconds=None, seconds=1, max_files=None,
                         windows_per_file=1, batch_size=1, workers=0, device="cpu",
                         calibrate_output=str(self.root / "calibrated.pt"), output=str(self.root / "report.json"))
        reports = []
        for full in (False, True):
            args.full_recordings = full
            with redirect_stdout(io.StringIO()):
                reports.append(evaluate(args))
        for report in reports:
            self.assertEqual(report["model_version"], "11")
            self.assertEqual(report["note_duration"], "sustain_extended")
            self.assertEqual(report["selection_mode"], "min")
            self.assertEqual(report["selection_metric"], "avgf_loss")
            self.assertEqual(report["avgf_betas"], [1, 2, 3, 4, 5])
            self.assertTrue(np.isfinite(report["avgf_loss"]))
            self.assertIn("note_macro_f12345", report)
        self.assertEqual(reports[0]["mode"], "interior_windows")
        self.assertEqual(reports[1]["mode"], "full_recordings")
        with (self.root / "report.pitches.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertTrue({"f4", "f5", "f12345", "in_macro_f12345"}.issubset(reader.fieldnames))
            self.assertEqual(len(list(reader)), 88)
        saved = torch.load(args.calibrate_output, weights_only=True)
        self.assertEqual(saved["format_version"], 13)
        self.assertEqual(saved["note_duration"], "sustain_extended")
        self.assertEqual(saved["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(load_model(args.calibrate_output, torch.device("cpu")).model_version, "11")

    def test_app_discovery_prefers_v11(self):
        from piano_ml.app import PianoApp
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        for version in ("8.1", "10.1", "11"):
            (checkpoints / f"piano-v{version}.pt").touch()
        app = SimpleNamespace(model_paths={}, models_box={}, model=Mock(), preview_thresholds=Mock())
        app.model.get.return_value = ""
        with patch("piano_ml.app.PROJECT", self.root):
            PianoApp.refresh_models(app)
        app.model.set.assert_called_once_with("piano-v11.pt")
        self.assertIn("piano-v10.1.pt", app.models_box["values"])
        app.preview_thresholds.assert_called_once()


if __name__ == "__main__":
    unittest.main()
