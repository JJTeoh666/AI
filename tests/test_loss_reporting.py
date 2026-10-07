import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from piano_ml.thresholds import Thresholds
from piano_ml.training import (_FrameBCEAccumulator, LOSS_COMPONENTS, pitch_loss_weights,
                               score, training_loss, training_loss_components)


class LossReportingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def predictions_and_targets(self):
        generator = torch.Generator().manual_seed(123)
        output = {head: torch.randn(2, 8, 88, generator=generator, requires_grad=True)
                  for head in ("frame", "onset", "offset", "velocity")}
        output["pedal"] = torch.randn(2, 8, 1, generator=generator, requires_grad=True)
        targets = {head: torch.zeros_like(value) for head, value in output.items()}
        targets["frame"][:, 1:5, 3] = 1
        targets["onset"][:, 1:3, 3] = 1
        targets["offset"][:, 5:7, 3] = 1
        targets["velocity"][:, 1:3, 3] = 0.7
        targets["pedal"][:, 2:5] = 1
        return output, targets

    def test_components_preserve_original_objective_and_gradients(self):
        for focused in (False, True):
            with self.subTest(focused=focused):
                output, targets = self.predictions_and_targets()
                weights = pitch_loss_weights(edge_loss_weight=2) if focused else torch.ones(88)
                # Reference the original weighted objective independently of the new helper.
                def bce(head, positive_weight):
                    errors = F.binary_cross_entropy_with_logits(
                        output[head], targets[head], pos_weight=torch.full((88,), positive_weight),
                        reduction="none")
                    return (errors * weights).sum() / (weights.sum() * errors.numel() / 88)
                expected = bce("frame", 5) + bce("onset", 10) + bce("offset", 10)
                boundary = F.max_pool1d(targets["offset"].transpose(1, 2), 5, 1, 2).transpose(1, 2) * weights
                frame_errors = F.binary_cross_entropy_with_logits(
                    output["frame"], targets["frame"], pos_weight=torch.full((88,), 5.0), reduction="none")
                expected += 0.25 * (frame_errors * boundary).sum() / boundary.sum().clamp_min(1)
                mask = targets["onset"] * weights
                expected += 0.5 * ((output["velocity"].sigmoid() - targets["velocity"]) ** 2 * mask).sum() / mask.sum().clamp_min(1)
                expected += 0.2 * F.binary_cross_entropy_with_logits(
                    output["pedal"], targets["pedal"], pos_weight=torch.tensor([2.0]))
                options = dict(offset_loss_weight=1, release_loss_weight=0.25,
                               pitch_weights=weights if focused else None)
                components = training_loss_components(output, targets, **options)
                actual = training_loss(output, targets, **options)
                self.assertEqual(tuple(components), LOSS_COMPONENTS[:-1])
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(sum(components.values()), actual, rtol=0, atol=0)
                before = torch.autograd.grad(expected, tuple(output.values()), retain_graph=True)
                after = torch.autograd.grad(actual, tuple(output.values()))
                for old_gradient, new_gradient in zip(before, after):
                    torch.testing.assert_close(new_gradient, old_gradient)

    def test_empty_onsets_and_releases_have_zero_finite_contributions(self):
        output, targets = self.predictions_and_targets()
        targets = {head: torch.zeros_like(value) for head, value in targets.items()}
        components = training_loss_components(output, targets, release_loss_weight=0.25,
                                             pitch_weights=pitch_loss_weights())
        self.assertEqual(float(components["release"].detach()), 0)
        self.assertEqual(float(components["velocity"].detach()), 0)
        sum(components.values()).backward()
        for value in output.values():
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(torch.equal(output["velocity"].grad, torch.zeros_like(output["velocity"])))
        _, with_releases = self.predictions_and_targets()
        disabled = training_loss_components(output, with_releases, release_loss_weight=0)
        self.assertEqual(float(disabled["release"]), 0)

    def test_frame_only_model_has_only_its_weighted_frame_contribution(self):
        output, targets = self.predictions_and_targets()
        components = training_loss_components(output["frame"], targets)
        expected = F.binary_cross_entropy_with_logits(output["frame"], targets["frame"],
                                                     pos_weight=torch.full((88,), 5.0))
        torch.testing.assert_close(training_loss(output["frame"], targets), expected)
        torch.testing.assert_close(components["frame"], expected)
        self.assertTrue(all(float(value) == 0 for name, value in components.items() if name != "frame"))

    def test_frame_bce_uses_element_counts_and_retains_no_gradients(self):
        first = torch.zeros(2, 3, 88, requires_grad=True)
        last = torch.full((1, 5, 88), 2.0, requires_grad=True)
        totals = _FrameBCEAccumulator()
        for logits in (first, last):
            totals.add(logits, torch.zeros_like(logits))
        combined = torch.cat((first.detach().flatten(), last.detach().flatten()))
        expected = F.binary_cross_entropy_with_logits(combined, torch.zeros_like(combined))
        self.assertAlmostEqual(totals.mean(), float(expected), places=6)
        self.assertFalse(totals.element_total.requires_grad)
        self.assertIsNone(totals.element_total.grad_fn)
        self.assertIsNone(totals.batch_total.grad_fn)
        self.assertIsNone(first.grad)
        self.assertIsNone(last.grad)

    def test_validation_matches_training_frame_bce_with_a_partial_final_batch(self):
        class FixedFrameModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, waves):
                self.calls += 1
                return waves[:, 0, None, None].expand(-1, 3, 88)

        waves = torch.zeros(3, 640)
        waves[-1] = 2
        targets = torch.zeros(3, 3, 88)
        loader = DataLoader(TensorDataset(waves, targets), batch_size=2)
        model = FixedFrameModel()
        report = score(model, loader, torch.device("cpu"), threshold_configs=[Thresholds()])
        training_frames = _FrameBCEAccumulator()
        legacy_means = []
        for audio, truth in loader:
            logits = audio[:, 0, None, None].expand_as(truth)
            training_frames.add(logits, truth)
            legacy_means.append(float(F.binary_cross_entropy_with_logits(logits, truth)))
        expected = F.binary_cross_entropy_with_logits(waves[:, 0, None, None].expand_as(targets), targets)
        self.assertEqual(model.calls, 2)
        self.assertFalse(model.training)
        self.assertAlmostEqual(report["frame_bce"], training_frames.mean(), places=12)
        self.assertAlmostEqual(report["frame_bce"], float(expected), places=6)
        self.assertAlmostEqual(report["loss"], sum(legacy_means) / len(legacy_means), places=12)
        self.assertNotAlmostEqual(report["frame_bce"], report["loss"], places=4)


if __name__ == "__main__":
    unittest.main()
