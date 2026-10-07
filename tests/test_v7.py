import csv
import io
import json
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
from piano_ml.data import MaestroWindows
from piano_ml.evaluation import evaluate
from piano_ml.inference import load_model, transcribe_audio
from piano_ml.learned_thresholds import PitchBalancedThresholds
from piano_ml.metrics import pitch_note_report
from piano_ml.model import BalancedPianoNet, MultiResolutionPianoNet, build_model
from piano_ml.thresholds import Thresholds
from piano_ml.training import pitch_loss_weights, train, training_loss, transfer_weights


def midi_bytes(pitch):
    events = bytes([0x81, 0x40, 0x90, pitch, 100, 0x83, 0x60, 0x80, pitch, 0, 0, 0xFF, 0x2F, 0])
    return (b"MThd" + (6).to_bytes(4, "big") + bytes([0, 0, 0, 1, 1, 0xE0])
            + b"MTrk" + len(events).to_bytes(4, "big") + events)


class V7Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        rows = []
        for name, split, pitch in (("bass", "train", 24), ("treble", "train", 96),
                                   ("middle", "train", 60), ("validation", "validation", 24),
                                   ("test", "test", 96)):
            (self.root / f"{name}.midi").write_bytes(midi_bytes(pitch))
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
        values = dict(data=str(self.root), output=str(output), architecture="onsets-multires-balanced", epochs=1,
                      batch_size=3, seconds=1.0, windows_per_file=1, max_files=None,
                      positive_weight=None, lr=1e-3, seed=42, workers=0, device="cpu",
                      resume=None, init_from=None, reset_optimizer=False, augment=False,
                      hidden_size=16, gru_layers=1)
        return Namespace(**{**values, **changes})

    def test_macro_f03_gives_a_rare_pitch_equal_weight_and_accumulates_counts(self):
        counts = np.zeros((88, 3), np.int64)
        counts[60 - 21] = (100, 0, 0)
        counts[24 - 21] = (1, 0, 9)
        counts[96 - 21] = (0, 2, 1)
        counts[108 - 21] = (0, 5, 0)
        report = pitch_note_report(counts)
        self.assertEqual(report["note_macro_pitch_count"], 3)
        self.assertAlmostEqual(report["note_macro_f0"], 2 / 3)
        self.assertAlmostEqual(report["note_macro_f3"], (1 + 10 / 91) / 3)
        self.assertAlmostEqual(report["note_macro_f03"],
                               (report["note_macro_f0"] + report["note_macro_f3"]) / 2)
        self.assertFalse(report["per_pitch_notes"][-1]["in_macro_f03"])
        self.assertEqual(report["per_pitch_notes"][-1]["fp"], 5)
        self.assertEqual(pitch_note_report(np.zeros((88, 3), np.int64))["note_macro_f03"], 0)
        # Candidate 1 has higher F0/F3 despite a lower F1.
        calibration = Calibration([Thresholds(frame=0.4), Thresholds(frame=0.6)])
        first, second = calibration.grid
        calibration.pitch_counts[first][24 - 21] = (1, 0, 9)
        calibration.pitch_counts[second][24 - 21] = (5, 5, 5)
        calibration.counts[first]["note"][:] = (1, 0, 9)
        calibration.counts[second]["note"][:] = (5, 5, 5)
        self.assertGreater(calibration.results("note_f1")["threshold"],
                           calibration.results("note_macro_f03")["threshold"])

    def test_bass_treble_and_ordinary_windows_and_fixed_validation(self):
        data = MaestroWindows(self.root, "train", seconds=1, windows_per_file=1,
                              random_windows=True, multi_target=True,
                              bass_sampling=0.3, treble_sampling=0.3)
        for draw, pitch in ((0.1, 24), (0.4, 96), (0.9, 60)):
            with patch("piano_ml.data.random.random", return_value=draw):
                _, targets = data[2]
            self.assertEqual(targets["reference_notes"][0]["pitch"], pitch)
        val = MaestroWindows(self.root, "validation", multi_target=True, seconds=1)
        self.assertEqual(val[0][1]["reference_notes"], val[0][1]["reference_notes"])
        with self.assertRaisesRegex(ValueError, "only available"):
            MaestroWindows(self.root, "validation", random_windows=True, treble_sampling=0.3)
        with self.assertRaisesRegex(ValueError, "sum"):
            MaestroWindows(self.root, "train", random_windows=True, bass_sampling=0.8, treble_sampling=0.3)

    def test_edge_loss_emphasizes_both_missed_and_false_positive_keys(self):
        weights = pitch_loss_weights(edge_loss_weight=2)
        for target_value in (0.0, 1.0):
            output = {head: torch.zeros(1, 10, 88, requires_grad=True)
                      for head in ("frame", "onset", "offset", "velocity")}
            output["pedal"] = torch.zeros(1, 10, 1, requires_grad=True)
            targets = {head: torch.full_like(logits, target_value) for head, logits in output.items()}
            loss = training_loss(output, targets, pitch_weights=weights)
            loss.backward()
            for head in ("frame", "onset", "offset"):
                middle = output[head].grad[..., 60 - 21].abs().mean()
                self.assertAlmostEqual(float(output[head].grad[..., 24 - 21].abs().mean() / middle), 2)
                self.assertAlmostEqual(float(output[head].grad[..., 96 - 21].abs().mean() / middle), 2)
            neutral = training_loss({key: value.detach() for key, value in output.items()}, targets,
                                    pitch_weights=torch.ones(88))
            original = training_loss({key: value.detach() for key, value in output.items()}, targets)
            self.assertTrue(torch.allclose(neutral, original))

    def test_threshold_gradients_are_continuous_and_do_not_change_logits(self):
        module = PitchBalancedThresholds()
        output = {head: torch.linspace(-3, 3, 12 * 88).reshape(1, 12, 88).requires_grad_()
                  for head in ("frame", "onset", "offset")}
        truth = {head: torch.zeros_like(value) for head, value in output.items()}
        for target in truth.values():
            target[:, 2:5, 24 - 21] = 1
            target[:, 6:8, 96 - 21] = 1
        before = module.values().detach().clone()
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.003, weight_decay=0)
        loss = module.loss(output, truth, regularization=0, pitch_weights=pitch_loss_weights())
        loss.backward()
        self.assertTrue(torch.all(module.raw.grad != 0))
        self.assertTrue(all(logits.grad is None for logits in output.values()))
        optimizer.step()
        after = module.values().detach()
        self.assertTrue(torch.all(after != before))
        self.assertTrue(torch.all((after > 0.05) & (after < 0.95)))
        self.assertEqual(module.raw.numel(), 3)
        self.assertFalse(module.export().pitch_dependent)
        optimizer.zero_grad()
        module.loss(output, {head: torch.zeros_like(value) for head, value in output.items()},
                    regularization=0).backward()
        self.assertTrue(torch.all(module.raw.grad < 0))  # Raise thresholds on empty targets.

    def test_v6_transfer_preserves_all_heads_and_precise_thresholds(self):
        source = MultiResolutionPianoNet(hidden_size=16, gru_layers=1).eval()
        upgraded = BalancedPianoNet(hidden_size=16, gru_layers=1).eval()
        self.assertTrue(transfer_weights(upgraded, {"model": source.state_dict()}))
        audio = torch.randn(1, 16000) * 0.1
        with torch.no_grad():
            old, new = source(audio), upgraded(audio)
        for head in old:
            self.assertTrue(torch.equal(old[head], new[head]), head)
        self.assertEqual(source.learned_thresholds(), upgraded.learned_thresholds())
        self.assertIsInstance(build_model(upgraded.config), BalancedPianoNet)

    def test_v7_training_threshold_updates_auto_resume_and_evaluation_csv(self):
        path = self.root / "v7.pt"
        initial = BalancedPianoNet(hidden_size=16, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": initial.state_dict(), "model_config": initial.config}, source)
        with redirect_stdout(io.StringIO()):
            train(self.options(path, init_from=str(source)))
        latest = self.root / "v7.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["model_config"]["architecture"], "onsets-multires-balanced")
        self.assertEqual(saved["format_version"], 8)
        self.assertEqual(saved["selection_metric"], "note_macro_f03")
        self.assertEqual(saved["training_config"]["patience"], 0)
        self.assertEqual(saved["training_config"]["threshold_calibration"], "training")
        self.assertEqual(saved["training_config"]["threshold_objective"], "pitch_f03")
        for key, value in (("bass_sampling", 0), ("treble_sampling", 0), ("middle_sampling", 0),
                           ("edge_loss_weight", 1), ("middle_loss_weight", 1)):
            self.assertEqual(saved["training_config"][key], value)
        self.assertEqual(saved["validation"]["calibration_candidates"], 1)
        self.assertEqual(saved["validation"]["calibration_search"], "parameters")
        self.assertTrue(saved["optimizer"]["param_groups"][1]["params"])
        self.assertFalse(torch.equal(initial.threshold_module.raw, saved["model"]["threshold_module.raw"]))
        history = json.loads(path.with_suffix(".history.jsonl").read_text().strip())
        self.assertGreater(history["threshold_loss"], 0)
        with redirect_stdout(io.StringIO()):
            train(self.options(path, architecture="auto", resume=str(latest), hidden_size=None, gru_layers=None, lr=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        threshold_id = resumed["optimizer"]["param_groups"][1]["params"][0]
        self.assertEqual(float(resumed["optimizer"]["state"][threshold_id]["step"]), 2)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu")
        self.assertEqual(result["architecture"], "onsets-multires-balanced")
        self.assertEqual(result["thresholds"], load_model(latest, torch.device("cpu")).learned_thresholds().to_dict())
        args = Namespace(data=str(self.root), checkpoint=str(latest), split="validation",
                         threshold=None, thresholds=None, selection_metric="auto", full_recordings=False,
                         min_note_seconds=None, seconds=1, max_files=None, windows_per_file=1,
                         batch_size=1, workers=0, device="cpu", calibrate_output=None,
                         output=str(self.root / "report.json"))
        with redirect_stdout(io.StringIO()):
            report = evaluate(args)
        self.assertEqual(report["selection_metric"], "note_macro_f03")
        with (self.root / "report.pitches.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 88)
        self.assertIn("f0", rows[0])
        self.assertIn("f3", rows[0])

    def test_checkpoint_follows_f03_and_training_continues_when_f1_improves(self):
        path = self.root / "selection.pt"
        calls = []
        def scores(model, *args, selection_metric, threshold_configs, **kwargs):
            self.assertEqual(selection_metric, "note_macro_f03")
            self.assertIsNone(threshold_configs)
            calls.append(model.learned_thresholds())
            f03 = 0.6 if len(calls) == 1 else 0.5
            f1 = 0.1 + len(calls) * 0.01
            return {"loss": 0.1, "frame_bce": 0.1, "f1": f1, "note_f1": f1, "note_f_avg": f1,
                    "note_macro_f03": f03, "note_macro_f0": f03, "note_macro_f3": f03,
                    "onset_f1": f1, "precision": f1, "recall": f1,
                    "threshold": calls[-1].frame, "thresholds": calls[-1].to_dict()}
        with patch("piano_ml.training.score", side_effect=scores), redirect_stdout(io.StringIO()):
            train(self.options(path, epochs=25))
        best = torch.load(path, weights_only=True)
        latest = torch.load(self.root / "selection.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["best_score"], 0.6)
        self.assertEqual(latest["epoch"], 25)
        self.assertEqual(latest["stale_epochs"], 24)
        self.assertGreater(latest["validation"]["note_f1"], best["validation"]["note_f1"])

    def test_cli_v7_and_reject_discrete_calibration(self):
        with patch("sys.argv", ["piano_ml", "train", "--architecture", "onsets-multires-balanced",
                                "--selection-metric", "note_macro_f03", "--treble-sampling", "0.4",
                                "--edge-loss-weight", "3"]), patch("piano_ml.__main__.train") as run:
            main()
        args = run.call_args.args[0]
        self.assertEqual(args.treble_sampling, 0.4)
        self.assertEqual(args.edge_loss_weight, 3)
        with self.assertRaisesRegex(ValueError, "gradient"):
            train(self.options(self.root / "bad.pt", threshold_calibration="validation"))


if __name__ == "__main__":
    unittest.main()
