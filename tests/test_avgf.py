import unittest
import numpy as np
import torch

from piano_ml.avgf import AvgFLoss, event_targets
from piano_ml.calibration import Calibration
from piano_ml.learned_thresholds import PitchFourScoreThresholds
from piano_ml.thresholds import Thresholds


class AvgFTest(unittest.TestCase):
    def test_validation_matches_training_objective_and_is_batch_partition_independent(self):
        generator = torch.Generator().manual_seed(12)
        module = PitchFourScoreThresholds().double()
        module.initialize(Thresholds(frame=0.4, onset=0.5, offset=0.6))
        with torch.no_grad():
            module.raw.add_(0.3)
        for empty, all_supported in ((False, False), (True, False), (False, True)):
            output = {head: torch.randn(3, 9, 88, generator=generator, dtype=torch.float64)
                      for head in ("frame", "onset", "offset")}
            targets = {head: torch.zeros_like(value) for head, value in output.items()}
            if not empty:
                for value in targets.values():
                    value[:, 2:6, :88 if all_supported else 3] = 1
            expected = float(module.loss(output, targets, temperature=0.13, regularization=0.07).detach())
            values = module.export()
            for parts in (((0, 3),), ((0, 2), (2, 3)), ((0, 1), (1, 2), (2, 3))):
                accumulator = AvgFLoss(values, module.prior.numpy(), temperature=0.13, regularization=0.07)
                for start, end in parts:
                    accumulator.add({key: value[start:end].sigmoid().numpy() for key, value in output.items()},
                                    {key: value[start:end].numpy() for key, value in targets.items()})
                actual = accumulator.results()
                self.assertAlmostEqual(actual["avgf_loss"], expected, places=12)
                self.assertAlmostEqual(actual["avgf_loss"], actual["avgf_data_loss"] + actual["avgf_prior_loss"], places=12)

    def test_lower_avgf_loss_selects_thresholds_even_when_decoded_scores_are_higher(self):
        low, high = Thresholds(frame=0.4), Thresholds(frame=0.6)
        calibration = Calibration([low, high])
        calibration.pitch_counts[low][39] = (1, 0, 0)
        calibration.pitch_counts[high][39] = (0, 0, 1)
        metrics = {low: {"avgf_loss": 0.7}, high: {"avgf_loss": 0.3}}
        result = calibration.results("avgf_loss", metrics)
        self.assertEqual(result["threshold"], 0.6)
        self.assertEqual(result["avgf_loss"], 0.3)
        self.assertEqual(calibration.results("note_macro_f0123", metrics)["threshold"], 0.4)

    def test_full_recording_event_targets_match_two_frame_grid_and_skip_clipped_ends(self):
        targets = event_targets(np.zeros((50, 88), bool),
                               [{"pitch": 60, "start": 0.1, "end": 0.4},
                                {"pitch": 68, "start": 0.98, "end": 1.0}], 1.0)
        np.testing.assert_array_equal(np.flatnonzero(targets["onset"][:, 39]), [5, 6])
        np.testing.assert_array_equal(np.flatnonzero(targets["offset"][:, 39]), [20, 21])
        self.assertEqual(targets["onset"][-1, 47], 1)
        self.assertEqual(targets["offset"][:, 47].sum(), 0)


if __name__ == "__main__":
    unittest.main()
