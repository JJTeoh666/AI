"""V12.1 parallel recurrent and residual FAN training, sustain, evaluation and desktop integration."""

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
from piano_ml.model import (SplitFANSustainPianoNet, ParallelFANSustainPianoNet,
                            V12_ARCHITECTURE, V121_ARCHITECTURE, build_model)
from piano_ml.training import LOSS_COMPONENTS, train, training_loss_components
from tests import test_v11_pipeline as v11
from tests import test_v111_pipeline as v111
from tests import test_v8 as v8


class V121PipelineTest(unittest.TestCase):
    setUp = v11.V11PipelineTest.setUp
    tearDown = v8.V8PipelineTest.tearDown
    records = v111.V111PipelineTest.records
    score_sequence = v111.V111PipelineTest.score_sequence

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, output, **changes):
        return v8.V8PipelineTest.options(self, output, **{
            "architecture": V121_ARCHITECTURE, "feature_width": 16, "hidden_size": 8,
            "gru_layers": 1, "event_width": 16, "event_fan_layers": 1,
            "fourier_modes": None, "fourier_layers": None, **changes})

    def test_fresh_auto_versioned_files_sustain_metadata_components_and_exact_resume(self):
        previous = Path.cwd()
        console = io.StringIO()
        try:
            os.chdir(self.root)
            args = self.options(None, architecture="auto", gru_layers=None, lr=None,
                                event_width=None, event_fan_layers=None)
            args.output = None
            with patch("piano_ml.training.MaestroWindows", wraps=MaestroWindows) as datasets, \
                    redirect_stdout(console):
                train(args)
        finally:
            os.chdir(previous)
        best = self.root / "checkpoints/piano-v12.1.pt"
        latest = self.root / "checkpoints/piano-v12.1.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertTrue(best.is_file())
        self.assertEqual(saved["format_version"], 16)
        self.assertEqual(saved["model_version"], "12.1")
        self.assertEqual(saved["model_config"]["architecture"], V121_ARCHITECTURE)
        self.assertEqual(saved["model_config"]["short_fft"], 2048)
        self.assertEqual(saved["model_config"]["long_fft"], 8192)
        self.assertEqual(saved["model_config"]["gru_layers"], 2)
        self.assertEqual(saved["model_config"]["event_width"], 256)
        self.assertEqual(saved["model_config"]["event_fan_layers"], 2)
        self.assertNotIn("fan_window", saved["model_config"])
        self.assertNotIn("pedal_hidden_size", saved["model_config"])
        self.assertEqual(saved["note_duration"], "sustain_extended")
        self.assertEqual(saved["avgf_betas"], [1, 2, 3, 4, 5])
        config = saved["training_config"]
        self.assertEqual(config["threshold_objective"], "pitch_f12345")
        self.assertEqual(config["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(config["note_duration"], "sustain_extended")
        self.assertTrue(config["pedal_events"])
        self.assertFalse(config["early_stopping"])
        self.assertEqual(config["lr_patience"], 15)
        self.assertEqual(saved["optimizer"]["param_groups"][0]["lr"], .0003)
        self.assertEqual(saved["optimizer"]["param_groups"][1]["lr"], .003)
        self.assertEqual(saved["selection_metric"], "avgf_loss")
        self.assertEqual(saved["selection_mode"], "min")
        self.assertEqual(saved["validation"]["avgf_betas"], [1, 2, 3, 4, 5])
        for call in datasets.call_args_list:
            self.assertEqual(call.kwargs["note_duration"], "sustain_extended")
            self.assertTrue(call.kwargs["pedal_events"])
        record = self.records(best)[0]
        self.assertEqual(set(record["train_loss_components"]), set(LOSS_COMPONENTS))
        self.assertAlmostEqual(sum(record["train_loss_components"].values()), record["train_loss"], places=6)
        self.assertEqual(record["train_loss_components"], saved["training_metrics"]["loss_components"])
        self.assertEqual(record["train_frame_bce"], saved["training_metrics"]["frame_bce"])
        self.assertIn("Training V12.1", console.getvalue())
        loaded = load_model(latest, torch.device("cpu"))
        recreated = build_model(saved["model_config"]).eval()
        recreated.load_state_dict(saved["model"])
        with torch.no_grad():
            audio = torch.randn(1, 16001) * .1
            first, second = loaded(audio), recreated(audio)
        self.assertEqual(len(first), 7)
        for head in first:
            torch.testing.assert_close(first[head], second[head], atol=0, rtol=0)
        with redirect_stdout(console):
            train(self.options(best, architecture="auto", resume=str(latest), lr=None,
                               hidden_size=None, gru_layers=None, feature_width=None,
                               event_width=None, event_fan_layers=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["model_config"], saved["model_config"])
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["scheduler"]["last_epoch"], saved["scheduler"]["last_epoch"] + 1)
        threshold_id = resumed["optimizer"]["param_groups"][1]["params"][0]
        self.assertEqual(float(resumed["optimizer"]["state"][threshold_id]["step"]), 2)
        self.assertEqual(len(self.records(best)), 2)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu", overlap_blend="equal")
        self.assertEqual(result["model_version"], "12.1")
        self.assertEqual(result["note_duration"], "sustain_extended")
        self.assertEqual(result["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(result["inference_config"]["overlap_blend"], "equal")

    def test_cli_event_options_compatible_initialization_and_old_resume(self):
        with patch("sys.argv", ["piano_ml", "train"]), patch("piano_ml.__main__.train") as run:
            main()
        self.assertIsNone(run.call_args.args[0].event_width)
        self.assertIsNone(run.call_args.args[0].event_fan_layers)
        self.assertIsNone(run.call_args.args[0].lr_patience)
        with patch("sys.argv", ["piano_ml", "train", "--architecture", V121_ARCHITECTURE,
                                "--event-width", "32", "--event-fan-layers", "1",
                                "--residual-blocks", "2",
                                "--short-fft", "1024", "--long-fft", "4096"]), \
                patch("piano_ml.__main__.train") as run:
            main()
        args = run.call_args.args[0]
        self.assertEqual((args.event_width, args.event_fan_layers), (32, 1))
        self.assertEqual(args.residual_blocks, 2)
        self.assertEqual((args.short_fft, args.long_fft), (1024, 4096))
        for changes, message in (({"fan_window": 8192}, "fan-window"),
                                 ({"hidden_size": 0}, "GRU"),
                                 ({"gru_layers": 0}, "GRU"),
                                 ({"event_width": 0}, "event"),
                                 ({"event_width": 12}, "event"),
                                 ({"event_fan_layers": 0}, "event|FAN")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                train(self.options(self.root / "invalid.pt", **changes))
        source = self.root / "source.pt"
        old = SplitFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1,
                                     event_width=16, event_fan_layers=1)
        torch.save({"model": old.state_dict(), "model_config": old.config}, source)
        with self.assertRaisesRegex(ValueError, "V12.1"):
            train(self.options(self.root / "invalid.pt", init_from=str(source)))
        fresh = ParallelFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1,
                                        event_width=32, event_fan_layers=1)
        torch.save({"model": fresh.state_dict(), "model_config": fresh.config}, source)
        with self.assertRaisesRegex(ValueError, "V12.1"):
            train(self.options(self.root / "invalid-dimensions.pt", init_from=str(source),
                               event_width=64, event_fan_layers=None))
        with redirect_stdout(io.StringIO()):
            train(self.options(self.root / "initialized.pt", init_from=str(source),
                               hidden_size=None, feature_width=None, gru_layers=None,
                               event_width=None, event_fan_layers=None))
            train(self.options(self.root / "old-v12.pt", architecture=V12_ARCHITECTURE))
            train(self.options(self.root / "old-v12.pt", architecture="auto",
                               resume=str(self.root / "old-v12.last.pt"), lr=None,
                               hidden_size=None, feature_width=None, gru_layers=None,
                               event_width=None, event_fan_layers=None))
        self.assertEqual(torch.load(self.root / "initialized.last.pt", weights_only=True)["model_config"], fresh.config)
        legacy = torch.load(self.root / "old-v12.last.pt", weights_only=True)
        self.assertEqual(legacy["model_version"], "12")
        self.assertEqual(legacy["model_config"]["architecture"], V12_ARCHITECTURE)
        self.assertEqual(legacy["epoch"], 2)

    def test_sustain_targets_and_all_pedal_heads_contribute(self):
        physical = MaestroWindows(self.root, "validation", seconds=1, windows_per_file=1,
                                  multi_target=True)
        sustained = MaestroWindows(self.root, "validation", seconds=1, windows_per_file=1,
                                   multi_target=True, note_duration="sustain_extended", pedal_events=True)
        audio, labels = sustained[0]
        _, physical_labels = physical[0]
        self.assertEqual(labels["frame"][20, 39], 1)
        self.assertEqual(physical_labels["frame"][20, 39], 0)
        self.assertEqual(labels["offset"][40, 39], 1)
        self.assertEqual(physical_labels["offset"][15, 39], 1)
        self.assertAlmostEqual(labels["reference_notes"][0]["end"], .8)
        model = ParallelFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1,
                                        event_width=16, event_fan_layers=2)
        prediction = model(torch.from_numpy(audio)[None])
        targets = {head: torch.from_numpy(value)[None] for head, value in labels.items()
                   if head != "reference_notes"}
        self.assertEqual(set(prediction), set(targets))
        components = training_loss_components(prediction, targets, offset_loss_weight=1,
                                              release_loss_weight=.25)
        expected = .2 * F.binary_cross_entropy_with_logits(prediction["pedal"], targets["pedal"],
                                                          pos_weight=torch.tensor([2.]))
        for head in ("pedal_onset", "pedal_offset"):
            expected = expected + .2 * F.binary_cross_entropy_with_logits(
                prediction[head], targets[head], pos_weight=torch.tensor([10.]))
        torch.testing.assert_close(components["pedal"], expected)
        sum(components.values()).backward()
        for head in (model.pedal_head, model.pedal_onset_head, model.pedal_offset_head):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_window_full_evaluation_metadata_csv_and_calibrated_checkpoint(self):
        model = ParallelFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1,
                                        event_width=16, event_fan_layers=2)
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
            self.assertEqual(report["model_version"], "12.1")
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
        self.assertEqual(saved["format_version"], 16)
        self.assertEqual(saved["note_duration"], "sustain_extended")
        self.assertEqual(saved["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(load_model(args.calibrate_output, torch.device("cpu")).model_version, "12.1")

    def test_all_overlap_methods_preserve_seven_heads_zero_overlap_and_cancellation(self):
        path = self.root / "boundary.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(bytes(round(4.13 * 16000) * 2))
        model = ParallelFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1,
                                        event_width=16, event_fan_layers=2).eval()
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

    def test_app_discovery_prefers_v121_and_preserves_old_versions(self):
        from piano_ml.app import PianoApp
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        for version in ("8.1", "10.1", "11", "11.1", "12", "12.1"):
            (checkpoints / f"piano-v{version}.pt").touch()
        app = SimpleNamespace(model_paths={}, models_box={}, model=Mock(), preview_thresholds=Mock())
        app.model.get.return_value = ""
        with patch("piano_ml.app.PROJECT", self.root):
            PianoApp.refresh_models(app)
        app.model.set.assert_called_once_with("piano-v12.1.pt")
        for name in ("piano-v12.pt", "piano-v11.1.pt", "piano-v11.pt", "piano-v10.1.pt", "piano-v8.1.pt"):
            self.assertIn(name, app.models_box["values"])
        app.preview_thresholds.assert_called_once()

    def test_min_avgf_selection_fifteen_bad_epochs_and_no_early_stopping(self):
        output = self.root / "selection.pt"
        # The baseline plus 15 bad epochs trigger one reduction, then a lower loss wins.
        values = [.5] * 16 + [.4, .45, .45]
        with patch("piano_ml.training.score", side_effect=self.score_sequence(values)), \
                redirect_stdout(io.StringIO()):
            train(self.options(output, epochs=len(values)))
        history = self.records(output)
        self.assertEqual(len(history), len(values))
        self.assertTrue(all(record["lr"] == 1e-3 for record in history[:15]))
        self.assertTrue(all(record["lr"] == .5e-3 for record in history[15:]))
        self.assertEqual([record["stale_epochs"] for record in history[-4:]], [15, 0, 1, 2])
        best = torch.load(output, weights_only=True)
        self.assertEqual(best["epoch"], 17)
        self.assertEqual(best["best_score"], .4)
        self.assertEqual(best["selection_metric"], "avgf_loss")
        self.assertEqual(best["selection_mode"], "min")
        saved = torch.load(self.root / "selection.last.pt", weights_only=True)
        self.assertEqual(saved["epoch"], len(values))
        self.assertEqual(saved["training_config"]["lr_patience"], 15)
        self.assertFalse(saved["training_config"]["early_stopping"])


if __name__ == "__main__":
    unittest.main()
