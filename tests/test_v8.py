import csv
import io
import math
import tempfile
import unittest
import wave
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.__main__ import main
from piano_ml.calibration import Calibration
from piano_ml.evaluation import evaluate
from piano_ml.inference import load_model, transcribe_audio
from piano_ml.metrics import pitch_note_report
from piano_ml.model import (BalancedPianoNet, FourierAnalysisLayer, FourierMixing, FourierRecurrentPianoNet,
                            FourierTemporalBlock, build_model, default_selection_metric)
from piano_ml.thresholds import Thresholds
from piano_ml.training import pitch_loss_weights, train, training_loss


class FourierModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fourier_layer_keeps_selected_modes_and_has_trainable_gradients(self):
        layer = FourierMixing(2, modes=3, padding=0)
        with torch.no_grad():
            layer.dc_weight.copy_(torch.eye(2))
            layer.band_weight.zero_()
            layer.band_weight[..., 0].copy_(torch.eye(2).expand(2, 2, 2))
        times = torch.arange(64) * (2 * math.pi / 64)
        expected = torch.stack((torch.sin(2 * times) + 0.5, torch.sin(times) * 0.25 + 0.1), dim=-1)[None]
        sequence = expected.clone()
        sequence[0, :, 0] += 0.3 * torch.cos(12 * times)
        sequence.requires_grad_()
        result = layer(sequence)
        self.assertTrue(torch.allclose(result, expected, atol=2e-6))
        result.square().mean().backward()
        for gradient in (sequence.grad, layer.dc_weight.grad, layer.band_weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_fourier_blocks_handle_short_odd_and_even_frame_lengths(self):
        block = FourierTemporalBlock(8, modes=9, dropout=0)
        for length in (1, 2, 5, 32, 33):
            block.zero_grad()
            sequence = torch.randn(2, length, 8, requires_grad=True)
            result = block(sequence)
            self.assertEqual(result.shape, sequence.shape)
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(sequence.grad).all())

    def test_fan_layer_has_periodic_pairs_and_trains_both_projections(self):
        layer = FourierAnalysisLayer(8, 16)
        features = torch.randn(2, 9, 8, requires_grad=True)
        result = layer(features)
        self.assertEqual(result.shape, (2, 9, 16))
        self.assertTrue(torch.allclose(result[..., :4].square() + result[..., 4:8].square(),
                                       torch.ones(2, 9, 4), atol=1e-6))
        result.sum().backward()
        self.assertGreater(float(layer.periodic.weight.grad.abs().sum()), 0)
        self.assertGreater(float(layer.aperiodic.weight.grad.abs().sum()), 0)

    def test_default_model_has_ten_million_active_parameters_and_all_branches_learn(self):
        model = FourierRecurrentPianoNet()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 10012924)
        output = model(torch.randn(1, 16000) * 0.1)
        self.assertEqual(set(output), {"frame", "onset", "offset", "velocity", "pedal"})
        for head, logits in output.items():
            self.assertEqual(logits.shape, (1, 51, 1 if head == "pedal" else 88))
        truth = {head: torch.zeros_like(logits) for head, logits in output.items()}
        truth["frame"][:, 10:25, 3] = 1
        truth["frame"][:, 15:30, 75] = 1
        for pitch, first, last in ((3, 10, 25), (75, 15, 30)):
            truth["onset"][:, first:first + 2, pitch] = 1
            truth["offset"][:, last:last + 2, pitch] = 1
            truth["velocity"][:, first:first + 2, pitch] = 0.7
        weights = pitch_loss_weights()
        loss = training_loss(output, truth, offset_loss_weight=1, release_loss_weight=0.25, pitch_weights=weights)
        loss = loss + model.threshold_module.loss(output, truth, pitch_weights=weights)
        loss.backward()
        checked = (model.short_encoder.convolution[0].weight, model.long_encoder.convolution[0].weight,
                   model.fusion[0].periodic.weight, model.fusion[0].aperiodic.weight,
                   model.fourier[0].mlp[0].periodic.weight, model.fourier[0].spectral.dc_weight,
                   model.fourier[0].spectral.band_weight, model.fourier[1].spectral.band_weight,
                   model.temporal.weight_ih_l0, model.temporal.weight_hh_l1_reverse,
                   model.onset_head.weight, model.frame_head.weight, model.offset_refinement[0].weight,
                   model.velocity_head.weight, model.pedal_head.weight, model.threshold_module.raw)
        for parameter in checked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(model.threshold_module.betas, (0, 1, 2, 3))
        self.assertFalse(model.learned_thresholds().pitch_dependent)
        self.assertEqual(default_selection_metric(model.architecture), "note_macro_f0123")
        self.assertEqual(default_selection_metric(BalancedPianoNet.architecture), "note_macro_f03")

    def test_macro_four_scores_give_each_supported_key_equal_weight(self):
        counts = np.zeros((88, 3), np.int64)
        counts[60 - 21] = (100, 0, 0)
        counts[24 - 21] = (1, 2, 3)
        counts[96 - 21] = (0, 0, 1)
        counts[-1] = (0, 5, 0)
        report = pitch_note_report(counts)
        row = report["per_pitch_notes"][24 - 21]
        expected = (1 / 3, 2 / 7, 5 / 19, 10 / 39)
        for beta, score in enumerate(expected):
            self.assertAlmostEqual(row[f"f{beta}"], score)
            self.assertAlmostEqual(report[f"note_macro_f{beta}"], (1 + score) / 3)
        self.assertAlmostEqual(row["f0123"], sum(expected) / 4)
        self.assertAlmostEqual(report["note_macro_f0123"], (1 + sum(expected) / 4) / 3)
        self.assertFalse(report["per_pitch_notes"][-1]["in_macro_f0123"])
        self.assertEqual(pitch_note_report(np.zeros((88, 3), np.int64))["note_macro_f0123"], 0)
        calibration = Calibration([Thresholds(frame=0.4), Thresholds(frame=0.6)])
        first, second = calibration.grid
        calibration.pitch_counts[first][24 - 21] = (1, 0, 9)
        calibration.pitch_counts[second][24 - 21] = (5, 5, 5)
        # F0/F3 favors precise but mostly missed notes; F0..F3 favors the second.
        self.assertEqual(calibration.results("note_macro_f03")["threshold"], 0.4)
        self.assertEqual(calibration.results("note_macro_f0123")["threshold"], 0.6)


class V8PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        rows = []
        for name, split, pitch in (("bass", "train", 24), ("treble", "train", 96),
                                   ("validation", "validation", 60), ("test", "test", 96)):
            events = bytes([0x81, 0x40, 0x90, pitch, 100, 0x83, 0x60, 0x80, pitch, 0, 0, 0xFF, 0x2F, 0])
            midi = (b"MThd" + (6).to_bytes(4, "big") + bytes([0, 0, 0, 1, 1, 0xE0])
                    + b"MTrk" + len(events).to_bytes(4, "big") + events)
            (self.root / f"{name}.midi").write_bytes(midi)
            with wave.open(str(self.root / f"{name}.wav"), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(np.zeros(16000, dtype="<i2").tobytes())
            rows.append({"split": split, "audio_filename": f"{name}.wav",
                         "midi_filename": f"{name}.midi", "duration": "1.0"})
        with (self.root / "maestro-v3.0.0.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def options(self, output, **changes):
        values = dict(data=str(self.root), output=str(output), architecture="auto", epochs=1,
                      batch_size=2, seconds=1.0, windows_per_file=1, max_files=None,
                      positive_weight=None, lr=1e-3, seed=42, workers=0, device="cpu",
                      resume=None, init_from=None, reset_optimizer=False, augment=False,
                      hidden_size=16, gru_layers=1, feature_width=32, fourier_modes=3, fourier_layers=1)
        return Namespace(**{**values, **changes})

    def test_auto_training_resume_checkpoint_roundtrip_and_app_inference(self):
        checkpoint = self.root / "v8.pt"
        with redirect_stdout(io.StringIO()):
            train(self.options(checkpoint))
        latest = self.root / "v8.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["format_version"], 9)
        self.assertEqual(saved["model_config"]["architecture"], "onsets-fourier-recurrent")
        self.assertEqual(saved["selection_metric"], "note_macro_f0123")
        self.assertEqual(saved["training_config"]["threshold_objective"], "pitch_f0123")
        self.assertEqual(saved["training_config"]["threshold_calibration"], "training")
        self.assertEqual(saved["training_config"]["patience"], 20)
        self.assertEqual(saved["training_config"]["treble_sampling"], 0.3)
        self.assertEqual(saved["validation"]["calibration_candidates"], 1)
        self.assertEqual(saved["validation"]["calibration_search"], "parameters")
        loaded = load_model(latest, torch.device("cpu"))
        recreated = build_model(saved["model_config"]).eval()
        recreated.load_state_dict(saved["model"])
        audio = torch.randn(1, 16320) * 0.1
        with torch.no_grad():
            before, after = loaded(audio), recreated(audio)
        for head in before:
            self.assertTrue(torch.equal(before[head], after[head]))
        with redirect_stdout(io.StringIO()):
            train(self.options(checkpoint, resume=str(latest), lr=None, hidden_size=None, gru_layers=None,
                               feature_width=None, fourier_modes=None, fourier_layers=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["model_config"], saved["model_config"])
        threshold_id = resumed["optimizer"]["param_groups"][1]["params"][0]
        self.assertEqual(float(resumed["optimizer"]["state"][threshold_id]["step"]), 2)
        self.assertFalse(torch.equal(saved["model"]["threshold_module.raw"], resumed["model"]["threshold_module.raw"]))
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu")
        self.assertEqual(result["architecture"], "onsets-fourier-recurrent")
        self.assertEqual(len(result["thresholds"]), 3)
        self.assertNotIn("register_thresholds", result)

    def test_best_checkpoint_scheduler_and_stopping_follow_four_score_average(self):
        checkpoint = self.root / "selection.pt"
        calls = []
        def scores(model, *args, selection_metric, threshold_configs, **kwargs):
            self.assertEqual(selection_metric, "note_macro_f0123")
            self.assertIsNone(threshold_configs)
            calls.append(model.learned_thresholds())
            value = 0.65 if len(calls) == 1 else 0.6
            other = 0.1 + len(calls) * 0.1
            return {"loss": 0.1, "f1": other, "note_f1": other, "note_f_avg": other,
                    "note_macro_f03": other, "note_macro_f0": other, "note_macro_f2": other,
                    "note_macro_f3": other, "note_macro_f0123": value,
                    "onset_f1": other, "precision": other, "recall": other,
                    "threshold": calls[-1].frame, "thresholds": calls[-1].to_dict()}
        with patch("piano_ml.training.score", side_effect=scores), redirect_stdout(io.StringIO()):
            train(self.options(checkpoint, epochs=5, patience=2, lr_patience=0))
        best = torch.load(checkpoint, weights_only=True)
        latest = torch.load(self.root / "selection.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["best_score"], 0.65)
        self.assertEqual(latest["epoch"], 3)
        self.assertEqual(latest["stale_epochs"], 2)
        self.assertLess(latest["optimizer"]["param_groups"][0]["lr"], 1e-3)
        self.assertGreater(latest["validation"]["note_macro_f03"], best["validation"]["note_macro_f03"])

    def test_evaluation_csv_and_calibrated_copy_preserve_v8_format(self):
        model = FourierRecurrentPianoNet(feature_width=32, fourier_modes=3, fourier_layers=1,
                                         hidden_size=16, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        args = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                         threshold=None, thresholds=None, frame_threshold=0.55,
                         selection_metric="auto", full_recordings=False, min_note_seconds=None,
                         seconds=1, max_files=None, windows_per_file=1, batch_size=1, workers=0,
                         device="cpu", calibrate_output=str(self.root / "calibrated.pt"),
                         output=str(self.root / "report.json"))
        with redirect_stdout(io.StringIO()):
            report = evaluate(args)
        self.assertEqual(report["selection_metric"], "note_macro_f0123")
        self.assertEqual(torch.load(self.root / "calibrated.pt", weights_only=True)["format_version"], 9)
        self.assertAlmostEqual(load_model(self.root / "calibrated.pt", torch.device("cpu")).learned_thresholds().frame, 0.55)
        with (self.root / "report.pitches.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 88)
        self.assertTrue(all(key in rows[0] for key in ("f0", "f1", "f2", "f3", "f0123", "in_macro_f0123")))

    def test_v8_cli_dimensions_and_incompatible_initialization(self):
        with patch("sys.argv", ["piano_ml", "train", "--architecture", "onsets-fourier-recurrent",
                                "--feature-width", "384", "--fourier-modes", "9", "--fourier-layers", "2",
                                "--selection-metric", "note_macro_f0123"]), patch("piano_ml.__main__.train") as run:
            main()
        args = run.call_args.args[0]
        self.assertEqual((args.feature_width, args.fourier_modes, args.fourier_layers), (384, 9, 2))
        source = self.root / "old.pt"
        old = BalancedPianoNet(hidden_size=16, gru_layers=1)
        torch.save({"model": old.state_dict(), "model_config": old.config}, source)
        with self.assertRaisesRegex(ValueError, "new structure"):
            train(self.options(self.root / "bad.pt", init_from=str(source)))
        with self.assertRaisesRegex(ValueError, "gradient"):
            train(self.options(self.root / "bad.pt", threshold_calibration="validation"))
        with self.assertRaises(ValueError):
            FourierRecurrentPianoNet(fourier_modes=0)


if __name__ == "__main__":
    unittest.main()
