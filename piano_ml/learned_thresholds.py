"""Bounded learned thresholds for global heads and legacy piano registers."""

import torch
from torch import nn

from .thresholds import HEADS, REGISTERS, Thresholds


class RegisterThresholds(nn.Module):
    scope = "register"

    def __init__(self):
        super().__init__()
        initial = torch.tensor([[0.5] * 3, [0.5] * 3, [0.7] * 3])
        self.raw = nn.Parameter(torch.logit((initial - 0.05) / 0.9))
        self.register_buffer("prior", initial.clone())
        indices = [index for index, (_, low, high) in enumerate(REGISTERS)
                   for _ in range(low, high + 1)]
        self.register_buffer("pitch_groups", torch.tensor(indices), persistent=False)

    def values(self):
        # Bounds prevent runaway near-zero/one thresholds with vanishing gradients.
        return 0.05 + 0.9 * self.raw.sigmoid()

    def initialize(self, thresholds: Thresholds):
        initial = self.raw.new_tensor([[row[head] for row in thresholds.registers()] for head in HEADS])
        initial = initial.clamp(0.050001, 0.949999)
        with torch.no_grad():
            self.raw.copy_(torch.logit((initial - 0.05) / 0.9))
            self.prior.copy_(initial)

    def export(self) -> Thresholds:
        expanded = self.values()[:, self.pitch_groups].detach().cpu().tolist()
        return Thresholds(**dict(zip(HEADS, expanded)))

    def loss(self, output, targets, temperature=0.1, regularization=0.05):
        """Smooth F1 by head/register; classifier gradients use the original losses.

        Detaching classifier probabilities keeps threshold optimization from being
        absorbed into output-layer biases. Empty registers retain their prior.
        This frame-target surrogate is not the exact decoded note F1 metric.
        """
        if temperature <= 0 or regularization < 0:
            raise ValueError("Threshold temperature must be positive and regularization nonnegative.")
        values = self.values()
        losses = []
        for head_index, head in enumerate(HEADS):
            probabilities = output[head].detach().sigmoid()
            for group in range(len(REGISTERS)):
                mask = self.pitch_groups == group
                truth = targets[head][..., mask]
                if not bool(truth.any()):
                    continue
                decisions = torch.sigmoid((probabilities[..., mask] - values[head_index, group]) / temperature)
                matched = (decisions * truth).sum()
                f1 = 2 * matched / (decisions.sum() + truth.sum()).clamp_min(1e-6)
                losses.append(1 - f1)
        loss = torch.stack(losses).mean() if losses else values.sum() * 0
        return loss + regularization * ((values - self.prior) ** 2).mean()


class GlobalThresholds(nn.Module):
    """Three trainable cutoffs, one per head, shared by all 88 pitches."""

    scope = "global"

    def __init__(self):
        super().__init__()
        initial = torch.empty(len(HEADS)).uniform_(0.35, 0.65)
        self.raw = nn.Parameter(torch.logit((initial - 0.05) / 0.9))
        self.register_buffer("prior", initial.clone())

    def values(self):
        return 0.05 + 0.9 * self.raw.sigmoid()

    def initialize(self, thresholds: Thresholds):
        initial = self.raw.new_tensor(list(thresholds.summary().values())).clamp(0.050001, 0.949999)
        with torch.no_grad():
            self.raw.copy_(torch.logit((initial - 0.05) / 0.9))
            self.prior.copy_(initial)

    def export(self) -> Thresholds:
        return Thresholds(**dict(zip(HEADS, self.values().detach().cpu().tolist())))

    def loss(self, output, targets, temperature=0.1, regularization=0.05):
        if temperature <= 0 or regularization < 0:
            raise ValueError("Threshold temperature must be positive and regularization nonnegative.")
        values = self.values()
        losses = []
        for index, head in enumerate(HEADS):
            truth = targets[head]
            if not bool(truth.any()):
                continue
            decisions = torch.sigmoid((output[head].detach().sigmoid() - values[index]) / temperature)
            matched = (decisions * truth).sum()
            f1 = 2 * matched / (decisions.sum() + truth.sum()).clamp_min(1e-6)
            losses.append(1 - f1)
        loss = torch.stack(losses).mean() if losses else values.sum() * 0
        return loss + regularization * ((values - self.prior) ** 2).mean()


class PitchBalancedThresholds(GlobalThresholds):
    """V7: continuous global cutoffs optimized with a per-pitch F0/F3 surrogate."""

    betas = (0, 3)

    def loss(self, output, targets, temperature=0.1, regularization=0.05,
             pitch_weights=None):
        if temperature <= 0 or regularization < 0:
            raise ValueError("Threshold temperature must be positive and regularization nonnegative.")
        values = self.values()
        losses = []
        for index, head in enumerate(HEADS):
            truth = targets[head]
            decisions = torch.sigmoid((output[head].detach().sigmoid() - values[index]) / temperature)
            dimensions = tuple(range(truth.ndim - 1))
            matched = (decisions * truth).sum(dim=dimensions)
            positives = truth.sum(dim=dimensions)
            predicted = decisions.sum(dim=dimensions)
            supported = positives > 0
            weights = torch.ones_like(positives) if pitch_weights is None else pitch_weights
            if bool(supported.any()):
                # Soft TP/FP/FN are summed per key across this training batch.
                scores = torch.stack([(1 + beta ** 2) * matched
                    / (predicted + beta ** 2 * positives).clamp_min(1e-6)
                    for beta in self.betas]).mean(dim=0)
                losses.append(((1 - scores[supported]) * weights[supported]).sum()
                              / weights[supported].sum())
            if bool((~supported).any()):
                # Empty keys still penalize false activity, which an F-beta
                # numerator of zero cannot provide a threshold gradient for.
                activity = decisions.mean(dim=dimensions)
                losses.append((activity[~supported] * weights[~supported]).sum()
                              / weights[~supported].sum())
        loss = torch.stack(losses).mean() if losses else values.sum() * 0
        return loss + regularization * ((values - self.prior) ** 2).mean()


class PitchFourScoreThresholds(PitchBalancedThresholds):
    """V8: optimize continuous cutoffs for mean per-pitch F0, F1, F2 and F3."""

    betas = (0, 1, 2, 3)
