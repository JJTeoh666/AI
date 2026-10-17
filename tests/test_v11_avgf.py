import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from piano_ml.avgf import AvgFLoss, avgf_accumulators
from piano_ml.calibration import Calibration
from piano_ml.learned_thresholds import PitchFiveScoreThresholds, PitchFourScoreThresholds
from piano_ml.metrics import pitch_note_report
from piano_ml.model import FOURIER_ARCHITECTURE, V11_ARCHITECTURE
from piano_ml.thresholds import HEADS, Thresholds


class V11AvgFTest(unittest.TestCase):
    def test_known_precision_recall_f12345_and_legacy_scores(self):
        # This pitch has precision 0.5 and recall 0.25.
        counts = np.zeros((88, 3), dtype=np.int64)
        counts[39] = (2, 2, 6)
        counts[42] = (0, 3, 0)  # Unsupported keys are not in decoded macro scores.
        report = pitch_note_report(counts, include_f12345=True)
        row = report["per_pitch_notes"][39]
        expected = [(1 + beta ** 2) * 2 / ((1 + beta ** 2) * 2 + 2 + beta ** 2 * 6)
                    for beta in range(1, 6)]
        for beta, score in enumerate(expected, 1):
            self.assertAlmostEqual(row[f"f{beta}"], score)
        self.assertAlmostEqual(report["note_macro_f12345"], sum(expected) / 5)
        self.assertEqual(report["note_macro_pitch_count"], 1)
        self.assertFalse(report["per_pitch_notes"][42]["in_macro_f12345"])
        legacy = pitch_note_report(counts)
        self.assertNotIn("note_macro_f12345", legacy)
        self.assertNotIn("f5", legacy["per_pitch_notes"][39])
        self.assertEqual(legacy["note_macro_f0123"], report["note_macro_f0123"])
        self.assertEqual(legacy["note_macro_f1"], report["note_macro_f1"])

    def test_training_and_validation_match_for_partitioned_batches_and_soft_targets(self):
        generator = torch.Generator().manual_seed(110)
        module = PitchFiveScoreThresholds().double()
        module.initialize(Thresholds(frame=0.4, onset=0.5, offset=0.6))
        with torch.no_grad():
            module.raw.add_(0.2)
        for case in ("mixed", "empty", "all_supported", "soft"):
            with self.subTest(case=case):
                output = {head: torch.randn(3, 9, 88, generator=generator, dtype=torch.float64,
                                            requires_grad=True) for head in HEADS}
                targets = {head: torch.zeros_like(value) for head, value in output.items()}
                if case != "empty":
                    for value in targets.values():
                        value[:, 2:6, :88 if case == "all_supported" else 3] = 0.35 if case == "soft" else 1
                loss = module.loss(output, targets, temperature=0.13, regularization=0.07)
                module.zero_grad()
                loss.backward()
                self.assertTrue(torch.isfinite(module.raw.grad).all())
                self.assertTrue(all(value.grad is None for value in output.values()))
                for parts in (((0, 3),), ((0, 2), (2, 3)), ((0, 1), (1, 2), (2, 3))):
                    accumulator = AvgFLoss(module.export(), module.prior.numpy(), temperature=0.13,
                                           regularization=0.07, betas=module.betas)
                    for start, end in parts:
                        accumulator.add({key: value[start:end].detach().sigmoid().numpy()
                                         for key, value in output.items()},
                                        {key: value[start:end].numpy() for key, value in targets.items()})
                    metrics = accumulator.results()
                    self.assertAlmostEqual(metrics["avgf_loss"], float(loss.detach()), places=12)
                    self.assertEqual(metrics["avgf_betas"], [1, 2, 3, 4, 5])

    def test_validation_matches_hand_computed_supported_and_empty_key_terms(self):
        probabilities = {head: np.full((2, 88), 0.5) for head in HEADS}
        targets = {head: np.zeros((2, 88)) for head in HEADS}
        for truth in targets.values():
            truth[0, 0] = 1
            truth[:, 1] = 0.25
        accumulator = AvgFLoss(Thresholds(frame=0.5, onset=0.5, offset=0.5),
                               [0.4] * 3, temperature=0.1, regularization=0.1,
                               betas=(1, 2, 3, 4, 5))
        accumulator.add(probabilities, targets)
        metrics = accumulator.results()
        second_score = sum((1 + beta ** 2) * 0.25 / (1 + beta ** 2 * 0.5)
                           for beta in range(1, 6)) / 5
        supported_loss = ((1 - 0.5) + (1 - second_score)) / 2
        expected_data = (supported_loss + 0.5) / 2
        self.assertAlmostEqual(metrics["avgf_data_loss"], expected_data, places=12)
        self.assertAlmostEqual(metrics["avgf_prior_loss"], 0.001, places=12)
        self.assertAlmostEqual(metrics["avgf_loss"], expected_data + 0.001, places=12)

    def test_factory_uses_threshold_module_betas_and_legacy_default_is_unchanged(self):
        for cls, architecture in ((PitchFourScoreThresholds, FOURIER_ARCHITECTURE),
                                  (PitchFiveScoreThresholds, V11_ARCHITECTURE)):
            module = cls()
            model = SimpleNamespace(architecture=architecture, threshold_module=module)
            values = module.export()
            accumulator = avgf_accumulators(model, [values])[values]
            self.assertEqual(accumulator.betas, cls.betas)
        self.assertEqual(AvgFLoss(Thresholds(), [0.5] * 3).betas, (0, 1, 2, 3))
        self.assertEqual(PitchFourScoreThresholds.betas, (0, 1, 2, 3))
        self.assertEqual(PitchFiveScoreThresholds.betas, (1, 2, 3, 4, 5))

    def test_v11_calibration_reports_f12345_and_still_minimizes_avgf_loss(self):
        low, high = Thresholds(frame=0.4), Thresholds(frame=0.6)
        calibration = Calibration([low, high])
        calibration.pitch_counts[low][39] = (8, 1, 2)
        calibration.pitch_counts[high][39] = (2, 1, 8)
        extras = {low: {"avgf_loss": 0.7, "avgf_betas": [1, 2, 3, 4, 5]},
                  high: {"avgf_loss": 0.3, "avgf_betas": [1, 2, 3, 4, 5]}}
        report = calibration.results("avgf_loss", extras)
        self.assertEqual(report["threshold"], 0.6)
        self.assertIn("note_macro_f12345", report)
        self.assertIn("f5", report["per_pitch_notes"][39])
        self.assertIn("note_macro_f0123", report)
        # Legacy aggregate mean remains F0..F4, independent of V11's objective.
        self.assertEqual(report["note_f_avg"], 0)
        self.assertEqual(calibration.results("note_macro_f12345")["threshold"], 0.4)
        legacy = calibration.results("note_macro_f0123")
        self.assertNotIn("note_macro_f12345", legacy)

    def test_invalid_betas_are_rejected_and_empty_reports_are_finite(self):
        for betas in ((), (-1,), (math.nan,), (math.inf,)):
            with self.subTest(betas=betas), self.assertRaisesRegex(ValueError, "betas"):
                AvgFLoss(Thresholds(), [0.5] * 3, betas=betas)
        report = pitch_note_report(np.zeros((88, 3)), include_f12345=True)
        self.assertEqual(report["note_macro_f12345"], 0)


if __name__ == "__main__":
    unittest.main()
