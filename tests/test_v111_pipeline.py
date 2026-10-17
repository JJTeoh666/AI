"""V11.1 dual-STFT training integration and exact plateau-LR behavior."""

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

from piano_ml.__main__ import main
from piano_ml.data import MaestroWindows
from piano_ml.evaluation import evaluate
from piano_ml.inference import load_model, predict_probabilities, transcribe_audio
from piano_ml.model import (DualSTFTSustainPianoNet, STFTFANSustainPianoNet,
                            V111_ARCHITECTURE, V11_ARCHITECTURE, build_model)
from piano_ml.training import LOSS_COMPONENTS, make_lr_scheduler, train
from tests import test_v11_pipeline as v11
from tests import test_v8 as v8


class V111PipelineTest(unittest.TestCase):
    setUp = v11.V11PipelineTest.setUp
    tearDown = v8.V8PipelineTest.tearDown

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, output, **changes):
        return v8.V8PipelineTest.options(self, output, **{
            "architecture": V111_ARCHITECTURE, "feature_width": 16, "hidden_size": 8,
            "gru_layers": 1, "fourier_modes": None, "fourier_layers": None, **changes})

    def records(self, output):
        return [json.loads(line) for line in output.with_suffix(".history.jsonl").read_text().splitlines()]

    def score_sequence(self, values):
        values = iter(values)

        def score(model, *args, selection_metric, **kwargs):
            self.assertEqual(selection_metric, "avgf_loss")
            threshold = model.learned_thresholds()
            return {"loss": .1, "frame_bce": .1, "f1": .2, "note_f1": .2,
                    "note_f_avg": .2, "avgf_loss": next(values), "onset_f1": .2,
                    "precision": .2, "recall": .2, "threshold": threshold.frame,
                    "thresholds": threshold.to_dict()}

        return score

    def test_fresh_explicit_v111_files_metadata_sustain_losses_and_resume(self):
        previous = Path.cwd()
        console = io.StringIO()
        try:
            os.chdir(self.root)
            args = self.options(None, architecture=V111_ARCHITECTURE, gru_layers=None)
            args.output = None
            with patch("piano_ml.training.MaestroWindows", wraps=MaestroWindows) as datasets, \
                    redirect_stdout(console):
                train(args)
        finally:
            os.chdir(previous)
        best = self.root / "checkpoints/piano-v11.1.pt"
        latest = self.root / "checkpoints/piano-v11.1.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertTrue(best.is_file())
        self.assertEqual(saved["format_version"], 14)
        self.assertEqual(saved["model_version"], "11.1")
        self.assertEqual(saved["model_config"]["architecture"], V111_ARCHITECTURE)
        self.assertEqual(saved["model_config"]["short_fft"], 2048)
        self.assertEqual(saved["model_config"]["long_fft"], 8192)
        self.assertNotIn("fan_window", saved["model_config"])
        self.assertEqual(saved["model_config"]["gru_layers"], 4)
        self.assertEqual(saved["note_duration"], "sustain_extended")
        self.assertEqual(saved["avgf_betas"], [1, 2, 3, 4, 5])
        config = saved["training_config"]
        self.assertEqual(config["threshold_objective"], "pitch_f12345")
        self.assertEqual(config["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(config["note_duration"], "sustain_extended")
        self.assertTrue(config["pedal_events"])
        self.assertFalse(config["early_stopping"])
        self.assertEqual(config["lr_patience"], 15)
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
        self.assertIn("Training V11.1", console.getvalue())
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
                               hidden_size=None, gru_layers=None, feature_width=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["model_config"], saved["model_config"])
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["scheduler"]["last_epoch"], saved["scheduler"]["last_epoch"] + 1)
        threshold_id = resumed["optimizer"]["param_groups"][1]["params"][0]
        self.assertEqual(float(resumed["optimizer"]["state"][threshold_id]["step"]), 2)
        self.assertEqual(len(self.records(best)), 2)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu", overlap_blend="equal")
        self.assertEqual(result["model_version"], "11.1")
        self.assertEqual(result["note_duration"], "sustain_extended")
        self.assertEqual(result["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(result["inference_config"]["overlap_blend"], "equal")

    def test_cli_spectral_options_initialization_and_v11_resume_remain_separate(self):
        with patch("sys.argv", ["piano_ml", "train"]), patch("piano_ml.__main__.train") as run:
            main()
        self.assertIsNone(run.call_args.args[0].lr_patience)
        with patch("sys.argv", ["piano_ml", "train", "--architecture", V111_ARCHITECTURE,
                                "--short-fft", "1024", "--long-fft", "4096", "--lr-patience", "7"]), \
                patch("piano_ml.__main__.train") as run:
            main()
        args = run.call_args.args[0]
        self.assertEqual((args.short_fft, args.long_fft, args.lr_patience), (1024, 4096, 7))
        with self.assertRaisesRegex(ValueError, "fan-window"):
            train(self.options(self.root / "invalid.pt", fan_window=8192))
        source = self.root / "source.pt"
        old = STFTFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        torch.save({"model": old.state_dict(), "model_config": old.config}, source)
        with self.assertRaisesRegex(ValueError, "V11.1"):
            train(self.options(self.root / "invalid.pt", init_from=str(source)))
        fresh = DualSTFTSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        torch.save({"model": fresh.state_dict(), "model_config": fresh.config}, source)
        with redirect_stdout(io.StringIO()):
            train(self.options(self.root / "initialized.pt", init_from=str(source),
                               hidden_size=None, feature_width=None, gru_layers=None))
            train(self.options(self.root / "old-v11.pt", architecture=V11_ARCHITECTURE))
            train(self.options(self.root / "old-v11.pt", architecture="auto",
                               resume=str(self.root / "old-v11.last.pt"), lr=None,
                               hidden_size=None, feature_width=None, gru_layers=None))
        self.assertEqual(torch.load(self.root / "initialized.last.pt", weights_only=True)["model_config"], fresh.config)
        legacy = torch.load(self.root / "old-v11.last.pt", weights_only=True)
        self.assertEqual(legacy["model_version"], "11")
        self.assertEqual(legacy["model_config"]["architecture"], V11_ARCHITECTURE)
        self.assertEqual(legacy["model_config"]["fan_window"], 8192)
        self.assertEqual(legacy["epoch"], 2)

    def test_window_full_evaluation_uses_sustain_and_f12345_metadata(self):
        model = DualSTFTSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        args = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                         threshold=None, thresholds=None, frame_threshold=.55, selection_metric="auto",
                         full_recordings=False, min_note_seconds=None, seconds=1, max_files=None,
                         windows_per_file=1, batch_size=1, workers=0, device="cpu",
                         calibrate_output=str(self.root / "calibrated.pt"), output=str(self.root / "report.json"))
        for full in (False, True):
            args.full_recordings = full
            with redirect_stdout(io.StringIO()):
                report = evaluate(args)
            self.assertEqual(report["model_version"], "11.1")
            self.assertEqual(report["note_duration"], "sustain_extended")
            self.assertEqual(report["selection_mode"], "min")
            self.assertEqual(report["selection_metric"], "avgf_loss")
            self.assertEqual(report["avgf_betas"], [1, 2, 3, 4, 5])
            self.assertTrue(np.isfinite(report["avgf_loss"]))
            self.assertIn("note_macro_f12345", report)
        saved = torch.load(args.calibrate_output, weights_only=True)
        self.assertEqual(saved["format_version"], 14)
        self.assertEqual(saved["note_duration"], "sustain_extended")
        self.assertEqual(saved["avgf_betas"], [1, 2, 3, 4, 5])
        self.assertEqual(load_model(args.calibrate_output, torch.device("cpu")).model_version, "11.1")

    def test_all_overlap_methods_preserve_seven_heads_and_cancellation(self):
        path = self.root / "boundary.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(round(4.13 * 16000) * 2))
        model = DualSTFTSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1).eval()
        for mode in ("crop", "weighted", "equal"):
            with self.subTest(mode=mode):
                progress = []
                duration, outputs = predict_probabilities(
                    path, model, torch.device("cpu"), context_seconds=1, overlap_blend=mode,
                    progress=lambda done, total: progress.append((done, total)))
                self.assertAlmostEqual(duration, 4.13)
                self.assertEqual(len(outputs), 7)
                self.assertEqual(progress, [(1, 2), (2, 2)])
                for head, values in outputs.items():
                    self.assertEqual(values.shape, (207, 1 if head.startswith("pedal") else 88))
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(((values >= 0) & (values <= 1)).all())
        cancel = threading.Event()
        with self.assertRaises(InterruptedError):
            predict_probabilities(path, model, torch.device("cpu"), cancel=cancel,
                                  overlap_blend="weighted", progress=lambda *_: cancel.set())

    def test_app_discovery_prefers_v111_and_keeps_old_versions_available(self):
        from piano_ml.app import PianoApp
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        for version in ("8.1", "10.1", "11", "11.1"):
            (checkpoints / f"piano-v{version}.pt").touch()
        app = SimpleNamespace(model_paths={}, models_box={}, model=Mock(), preview_thresholds=Mock())
        app.model.get.return_value = ""
        with patch("piano_ml.app.PROJECT", self.root):
            PianoApp.refresh_models(app)
        app.model.set.assert_called_once_with("piano-v11.1.pt")
        for name in ("piano-v11.pt", "piano-v10.1.pt", "piano-v8.1.pt"):
            self.assertIn(name, app.models_box["values"])
        app.preview_thresholds.assert_called_once()

    def test_default_lr_halves_on_exactly_fifteenth_bad_epoch_and_runs_requested_epochs(self):
        output = self.root / "plateau.pt"
        with patch("piano_ml.training.score", side_effect=self.score_sequence([.5] * 19)), \
                redirect_stdout(io.StringIO()):
            train(self.options(output, epochs=19))
        history = self.records(output)
        self.assertEqual(len(history), 19)
        self.assertTrue(all(record["lr"] == 1e-3 for record in history[:15]))
        self.assertTrue(all(record["lr"] == .5e-3 for record in history[15:]))
        self.assertEqual(history[14]["stale_epochs"], 14)
        self.assertEqual(history[15]["stale_epochs"], 15)
        saved = torch.load(self.root / "plateau.last.pt", weights_only=True)
        self.assertEqual(saved["epoch"], 19)
        self.assertEqual(saved["training_config"]["lr_patience"], 15)
        self.assertFalse(saved["training_config"]["early_stopping"])
        self.assertEqual(torch.load(output, weights_only=True)["epoch"], 1)

    def test_lr_improvement_uses_same_absolute_tolerance_as_best_model(self):
        output = self.root / "tolerance.pt"
        values = [.5, .49999975, .499998, .49999775, .49999775, .49999775]
        with patch("piano_ml.training.score", side_effect=self.score_sequence(values)), \
                redirect_stdout(io.StringIO()):
            train(self.options(output, epochs=len(values), lr_patience=3))
        history = self.records(output)
        self.assertEqual([record["stale_epochs"] for record in history], [0, 1, 0, 1, 2, 3])
        self.assertEqual([record["lr"] for record in history], [1e-3] * 5 + [.5e-3])
        self.assertEqual(torch.load(output, weights_only=True)["epoch"], 3)

    def test_resume_preserves_saved_lr_patience_and_explicit_override_updates_it(self):
        output = self.root / "resume-patience.pt"
        latest = self.root / "resume-patience.last.pt"
        with patch("piano_ml.training.score", side_effect=self.score_sequence([.5] * 5)), \
                redirect_stdout(io.StringIO()):
            train(self.options(output, epochs=3, lr_patience=4))
            train(self.options(output, epochs=1, resume=str(latest), lr=None, lr_patience=None,
                               feature_width=None, hidden_size=None, gru_layers=None))
            preserved = torch.load(latest, weights_only=True)
            train(self.options(output, epochs=1, resume=str(latest), lr=None, lr_patience=2,
                               feature_width=None, hidden_size=None, gru_layers=None))
        self.assertEqual(preserved["training_config"]["lr_patience"], 4)
        self.assertEqual(preserved["optimizer"]["param_groups"][0]["lr"], 1e-3)
        self.assertEqual(preserved["stale_epochs"], 3)
        overridden = torch.load(latest, weights_only=True)
        self.assertEqual(overridden["training_config"]["lr_patience"], 2)
        self.assertEqual(overridden["optimizer"]["param_groups"][0]["lr"], .5e-3)
        self.assertEqual(overridden["epoch"], 5)
        self.assertEqual(overridden["stale_epochs"], 4)

    def test_legacy_resume_without_new_patience_fields_restores_old_scheduler_rule(self):
        output = self.root / "legacy-patience.pt"
        latest = self.root / "legacy-patience.last.pt"
        with patch("piano_ml.training.score", side_effect=self.score_sequence([.5] * 5)), \
                redirect_stdout(io.StringIO()):
            train(self.options(output, architecture=V11_ARCHITECTURE, epochs=3, lr_patience=4))
            saved = torch.load(latest, weights_only=True)
            saved["training_config"].pop("lr_patience", None)
            saved["training_config"].pop("lr_patience_semantics", None)
            saved["scheduler"].update(patience=3, threshold=1e-4, threshold_mode="rel")
            torch.save(saved, latest)
            train(self.options(output, architecture="auto", epochs=1, resume=str(latest), lr=None,
                               lr_patience=None, feature_width=None, hidden_size=None, gru_layers=None))
            first = torch.load(latest, weights_only=True)
            train(self.options(output, architecture="auto", epochs=1, resume=str(latest), lr=None,
                               lr_patience=None, feature_width=None, hidden_size=None, gru_layers=None))
        self.assertEqual(first["model_version"], "11")
        self.assertEqual(first["scheduler"]["patience"], 3)
        self.assertEqual(first["scheduler"]["threshold_mode"], "rel")
        self.assertEqual(first["scheduler"]["threshold"], 1e-4)
        self.assertEqual(first["optimizer"]["param_groups"][0]["lr"], 1e-3)
        final = torch.load(latest, weights_only=True)
        self.assertEqual(final["optimizer"]["param_groups"][0]["lr"], .5e-3)
        self.assertEqual(final["epoch"], 5)

    def test_resume_to_new_output_preserves_saved_patience_when_comparison_resets(self):
        output = self.root / "source-policy.pt"
        resumed_output = self.root / "new-policy.pt"
        with patch("piano_ml.training.score", side_effect=self.score_sequence([.5] * 3)), \
                redirect_stdout(io.StringIO()):
            train(self.options(output, lr_patience=7))
            train(self.options(resumed_output, architecture="auto",
                               resume=str(self.root / "source-policy.last.pt"), lr=None, lr_patience=None,
                               feature_width=None, hidden_size=None, gru_layers=None))
        resumed = torch.load(self.root / "new-policy.last.pt", weights_only=True)
        self.assertEqual(resumed["training_config"]["lr_patience"], 7)
        self.assertEqual(resumed["scheduler"]["patience"], 6)
        self.assertEqual(resumed["stale_epochs"], 1)
        self.assertEqual(resumed["scheduler"]["num_bad_epochs"], 1)
        self.assertEqual(resumed["epoch"], 2)

    def test_scheduler_exact_failure_count_for_min_and_max_and_no_repeated_drop(self):
        for mode in ("min", "max"):
            with self.subTest(mode=mode):
                optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1.)
                scheduler = make_lr_scheduler(optimizer, mode, patience=15)
                scheduler.step(.5)
                for _ in range(14):
                    scheduler.step(.5)
                    self.assertEqual(optimizer.param_groups[0]["lr"], 1.)
                scheduler.step(.5)
                self.assertEqual(optimizer.param_groups[0]["lr"], .5)
                for _ in range(14):
                    scheduler.step(.5)
                    self.assertEqual(optimizer.param_groups[0]["lr"], .5)


if __name__ == "__main__":
    unittest.main()
