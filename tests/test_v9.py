import io
import json
import math
import os
import threading
import unittest
import wave
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.__main__ import main
from piano_ml.evaluation import evaluate
from piano_ml.inference import load_model, predict_probabilities, transcribe_audio
from piano_ml.model import (V9_ARCHITECTURE, ThreeBranchPianoNet,
                            RawFANEncoder, ResidualTemporalCNN, build_model, default_selection_metric)
from piano_ml.training import train, training_loss
from tests import test_v8 as v8


class V9ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_budget_four_second_backward_and_every_parameter_gradient(self):
        model = ThreeBranchPianoNet()
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.assertEqual(count(model), 10168060)
        self.assertEqual(count(model.short_encoder) + count(model.long_encoder), 1770432)
        self.assertEqual(count(model.raw_encoder), 1034304)
        self.assertEqual(count(model.fusion), 443520)
        self.assertEqual(count(model.residual), 5318784)
        self.assertEqual(count(model.temporal), 1331712)
        self.assertEqual(default_selection_metric(V9_ARCHITECTURE), "avgf_loss")
        branches = {}
        handles = [getattr(model, name).register_forward_hook(
            lambda module, inputs, output, key=name: branches.update({key: output.shape}))
            for name in ("short_encoder", "long_encoder", "raw_encoder")]
        output = model(torch.randn(1, 64000) * 0.1)
        for handle in handles:
            handle.remove()
        self.assertEqual(set(branches.values()), {torch.Size((1, 201, 384))})
        truth = {head: torch.zeros_like(logits) for head, logits in output.items()}
        for head, logits in output.items():
            self.assertEqual(logits.shape, (1, 201, 1 if head == "pedal" else 88))
        for pitch, first, last in ((3, 10, 25), (40, 70, 120), (75, 150, 180)):
            truth["frame"][:, first:last, pitch] = 1
            truth["onset"][:, first:first + 2, pitch] = 1
            truth["offset"][:, last:last + 2, pitch] = 1
            truth["velocity"][:, first:first + 2, pitch] = 0.7
        truth["pedal"][:, 60:130] = 1
        loss = training_loss(output, truth, offset_loss_weight=1, release_loss_weight=0.25)
        loss = loss + model.threshold_module.loss(output, truth)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(model.threshold_module.betas, (0, 1, 2, 3))
        self.assertFalse(model.learned_thresholds().pitch_dependent)

    def test_alignment_for_short_and_odd_inputs(self):
        model = ThreeBranchPianoNet(feature_width=16, hidden_size=8, gru_layers=1).eval()
        with torch.no_grad():
            for samples in (1, 319, 320, 321, 16001):
                with self.subTest(samples=samples):
                    audio = torch.randn(1, samples) * 0.1
                    count = samples // 320 + 1
                    self.assertEqual(model.raw_encoder.raw_frames(audio).shape, (1, count, 2048))
                    self.assertEqual(model.short_mel(audio).shape[-1], count)
                    self.assertEqual(model.long_mel(audio).shape[-1], count)
                    for head, output in model(audio).items():
                        self.assertEqual(output.shape, (1, count, 1 if head == "pedal" else 88))
                        self.assertTrue(torch.isfinite(output).all())

    def test_raw_windows_preserve_centers_padding_and_amplitude(self):
        encoder = RawFANEncoder(16, dropout=0)
        wave = torch.arange(2401, dtype=torch.float32)[None]
        frames = encoder.raw_frames(wave)
        # Hann's midpoint is one: the first sample and every hop stay centered.
        self.assertTrue(torch.equal(frames[0, :, 1024], wave[0, ::320]))
        self.assertEqual(float(frames[0, 0, :1024].abs().sum()), 0)
        self.assertEqual(float(frames[0, -1, 1425:].abs().sum()), 0)
        self.assertTrue(torch.allclose(encoder.raw_frames(wave * 3), frames * 3, rtol=1e-6))
        self.assertIn("window", dict(encoder.named_buffers()))
        self.assertNotIn("window", dict(encoder.named_parameters()))

    def test_residual_skip_and_independent_blocks(self):
        block = ResidualTemporalCNN(16, dropout=0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        sequence = torch.randn(2, 5, 16)
        self.assertTrue(torch.equal(block(sequence), torch.nn.functional.gelu(sequence)))
        model = ThreeBranchPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        weights = [part.convolution[0].weight for part in model.residual]
        self.assertEqual(len({weight.data_ptr() for weight in weights}), 3)
        for width in (8, 17, 31, 0, 16.0):
            with self.assertRaisesRegex(ValueError, "divisible by eight"):
                ThreeBranchPianoNet(feature_width=width)


class V9PipelineTest(unittest.TestCase):
    # Reuse the tiny paired WAV/MIDI dataset, without inheriting V8-specific tests.
    setUp = v8.V8PipelineTest.setUp
    tearDown = v8.V8PipelineTest.tearDown

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, output, **changes):
        return v8.V8PipelineTest.options(self, output, **{
            "architecture": V9_ARCHITECTURE, "fourier_modes": None, "fourier_layers": None, **changes})

    def test_explicit_v9_files_loss_reporting_resume_and_v9_initialization(self):
        previous_directory = Path.cwd()
        console = io.StringIO()
        args = self.options(None)
        args.output = None
        try:
            os.chdir(self.root)
            with redirect_stdout(console):
                train(args)
        finally:
            os.chdir(previous_directory)
        best = self.root / "checkpoints/piano-v9.pt"
        latest = self.root / "checkpoints/piano-v9.last.pt"
        history = self.root / "checkpoints/piano-v9.history.jsonl"
        saved = torch.load(latest, weights_only=True)
        self.assertTrue(best.exists())
        self.assertEqual(saved["format_version"], 10)
        self.assertEqual(saved["model_version"], "9")
        self.assertEqual(saved["training_config"]["model_version"], "9")
        self.assertEqual(saved["model_config"]["architecture"], V9_ARCHITECTURE)
        self.assertNotIn("fourier_modes", saved["model_config"])
        self.assertEqual(saved["selection_metric"], "avgf_loss")
        self.assertEqual(saved["selection_mode"], "min")
        self.assertEqual(saved["training_config"]["threshold_objective"], "pitch_f0123")
        self.assertEqual(saved["training_config"]["threshold_calibration"], "training")
        for key, value in (("bass_sampling", 0), ("middle_sampling", 0), ("treble_sampling", 0),
                           ("edge_loss_weight", 1), ("middle_loss_weight", 1),
                           ("offset_loss_weight", 1), ("release_loss_weight", 0.25)):
            self.assertEqual(saved["training_config"][key], value)
        record = json.loads(history.read_text())
        self.assertEqual(record["model_version"], "9")
        self.assertAlmostEqual(sum(record["train_loss_components"].values()), record["train_loss"], places=6)
        self.assertEqual(saved["training_metrics"]["frame_bce"], record["train_frame_bce"])
        self.assertIn("frame_bce", saved["validation"])
        self.assertIn("avgf_loss", saved["validation"])
        for field in ("train_total_loss=", "train_frame_bce=", "val_frame_bce=", "train_loss_components:"):
            self.assertIn(field, console.getvalue())
        self.assertIn("Training V9", console.getvalue())
        loaded = load_model(latest, torch.device("cpu"))
        recreated = build_model(saved["model_config"]).eval()
        recreated.load_state_dict(saved["model"])
        with torch.no_grad():
            wave = torch.randn(1, 16001) * 0.1
            before, after = loaded(wave), recreated(wave)
        for head in before:
            self.assertTrue(torch.equal(before[head], after[head]))
        with redirect_stdout(console):
            train(self.options(best, resume=str(latest), lr=None, hidden_size=None,
                               gru_layers=None, feature_width=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["model_config"], saved["model_config"])
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["scheduler"]["last_epoch"], saved["scheduler"]["last_epoch"] + 1)
        threshold_id = resumed["optimizer"]["param_groups"][1]["params"][0]
        self.assertEqual(float(resumed["optimizer"]["state"][threshold_id]["step"]), 2)
        self.assertFalse(torch.equal(saved["model"]["threshold_module.raw"], resumed["model"]["threshold_module.raw"]))
        self.assertIn("rng", resumed)
        self.assertEqual(len(history.read_text().splitlines()), 2)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu", overlap_blend="equal")
        self.assertEqual(result["model_version"], "9")
        self.assertEqual(result["inference_config"]["overlap_blend"], "equal")
        # Initialization carries compatible recognizer tensors but restarts the epoch/optimizer.
        initialized = self.root / "initialized.pt"
        with redirect_stdout(console):
            train(self.options(initialized, init_from=str(latest), lr=1e-3))
        self.assertIn("Transferred complete recognizer", console.getvalue())
        self.assertEqual(torch.load(self.root / "initialized.last.pt", weights_only=True)["epoch"], 1)

    def test_avgf_selection_scheduler_and_no_early_stop(self):
        checkpoint = self.root / "selection.pt"
        calls = []
        def scores(model, *args, selection_metric, **kwargs):
            self.assertEqual(selection_metric, "avgf_loss")
            calls.append(1)
            threshold = model.learned_thresholds()
            value = 0.3 if len(calls) == 2 else 0.5
            diagnostic = len(calls) * 0.1
            return {"loss": 0.1, "frame_bce": 0.1, "f1": diagnostic, "note_f1": diagnostic,
                    "note_f_avg": diagnostic, "avgf_loss": value, "onset_f1": diagnostic,
                    "precision": diagnostic, "recall": diagnostic,
                    "threshold": threshold.frame, "thresholds": threshold.to_dict()}
        with patch("piano_ml.training.score", side_effect=scores), redirect_stdout(io.StringIO()):
            train(self.options(checkpoint, epochs=4, patience=1, lr_patience=0))
            train(self.options(checkpoint, resume=str(self.root / "selection.last.pt"),
                               epochs=2, lr=None, hidden_size=None, gru_layers=None, feature_width=None))
        best = torch.load(checkpoint, weights_only=True)
        latest = torch.load(self.root / "selection.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 2)
        self.assertEqual(best["best_score"], 0.3)
        self.assertEqual(latest["epoch"], 6)
        self.assertEqual(latest["stale_epochs"], 4)
        self.assertFalse(latest["training_config"]["early_stopping"])
        self.assertEqual(latest["scheduler"]["mode"], "min")
        self.assertLess(latest["optimizer"]["param_groups"][0]["lr"], 1e-3)

    def test_resume_restores_random_and_optimizer_states_exactly(self):
        full, split = self.root / "full.pt", self.root / "split.pt"
        with redirect_stdout(io.StringIO()):
            train(self.options(full, epochs=2, augment=True))
            train(self.options(split, epochs=1, augment=True))
            train(self.options(split, epochs=1, resume=str(self.root / "split.last.pt"),
                               lr=None, hidden_size=None, gru_layers=None, feature_width=None, augment=None))
        uninterrupted = torch.load(self.root / "full.last.pt", weights_only=True)
        resumed = torch.load(self.root / "split.last.pt", weights_only=True)
        for name, parameter in uninterrupted["model"].items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(parameter, resumed["model"][name]))
        self.assertTrue(torch.equal(uninterrupted["rng"]["torch"], resumed["rng"]["torch"]))
        self.assertEqual(uninterrupted["rng"]["python"], resumed["rng"]["python"])
        self.assertEqual(uninterrupted["rng"]["numpy"], resumed["rng"]["numpy"])
        self.assertEqual(uninterrupted["scheduler"], resumed["scheduler"])
        self.assertEqual(uninterrupted["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
        for index, state in uninterrupted["optimizer"]["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][index][name]))

    def test_v9_cli_rejects_v81_transfer_and_temporal_fourier_options(self):
        with patch("sys.argv", ["piano_ml", "train", "--architecture", V9_ARCHITECTURE,
                                "--feature-width", "384"]), patch("piano_ml.__main__.train") as run:
            main()
        self.assertEqual(run.call_args.args[0].architecture, V9_ARCHITECTURE)
        source = self.root / "v81.pt"
        model = v8.FourierRecurrentPianoNet(feature_width=32, hidden_size=16, gru_layers=1)
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        with self.assertRaisesRegex(ValueError, "V8.1 weights cannot transfer"):
            train(self.options(self.root / "bad.pt", init_from=str(source)))
        for option in ("fourier_modes", "fourier_layers"):
            with self.assertRaisesRegex(ValueError, "specific to V8.1"):
                train(self.options(self.root / "bad.pt", **{option: 2}))
        with self.assertRaisesRegex(ValueError, "gradient"):
            train(self.options(self.root / "bad.pt", threshold_calibration="validation"))

    def test_evaluation_metadata_calibrated_copy_and_full_recordings(self):
        model = ThreeBranchPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        self.assertEqual(load_model(source, torch.device("cpu")).model_version, "9")
        args = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                         threshold=None, thresholds=None, frame_threshold=0.55, selection_metric="auto",
                         full_recordings=False, min_note_seconds=None, seconds=1, max_files=None,
                         windows_per_file=1, batch_size=1, workers=0, device="cpu",
                         calibrate_output=str(self.root / "calibrated.pt"), output=str(self.root / "report.json"))
        with redirect_stdout(io.StringIO()):
            report = evaluate(args)
            args.full_recordings = True
            full = evaluate(args)
        self.assertEqual(report["model_version"], "9")
        self.assertEqual(report["selection_mode"], "min")
        self.assertEqual(full["selection_metric"], "avgf_loss")
        self.assertTrue(math.isfinite(full["avgf_loss"]))
        self.assertEqual(torch.load(args.calibrate_output, weights_only=True)["format_version"], 10)
        self.assertEqual(load_model(args.calibrate_output, torch.device("cpu")).model_version, "9")
        self.assertTrue((self.root / "report.pitches.csv").exists())

    def test_all_overlap_methods_use_v9_heads_and_continuous_grid(self):
        path = self.root / "boundary.wav"
        duration = 8.13
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(np.zeros(round(duration * 16000), dtype="<i2").tobytes())
        model = ThreeBranchPianoNet(feature_width=16, hidden_size=8, gru_layers=1).eval()
        zero_overlap = []
        for context in (0, 1):
            for mode in ("crop", "weighted", "equal"):
                with self.subTest(context=context, mode=mode):
                    measured, probabilities = predict_probabilities(path, model, torch.device("cpu"),
                                                                    context_seconds=context, overlap_blend=mode)
                    self.assertAlmostEqual(measured, duration)
                    for head, values in probabilities.items():
                        self.assertEqual(values.shape, (407, 1 if head == "pedal" else 88))
                        self.assertTrue(np.isfinite(values).all())
                        self.assertTrue(((values >= 0) & (values <= 1)).all())
                    if context == 0:
                        zero_overlap.append(probabilities)
        for other in zero_overlap[1:]:
            for head in zero_overlap[0]:
                np.testing.assert_array_equal(other[head], zero_overlap[0][head])
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            predict_probabilities(path, model, torch.device("cpu"), cancel=cancel, overlap_blend="equal")


if __name__ == "__main__":
    unittest.main()
