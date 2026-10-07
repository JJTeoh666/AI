"""Validation counterpart of V8.1's smooth threshold AvgF objective."""

import math
import numpy as np

from .thresholds import HEADS


class AvgFLoss:
    """Accumulate soft counts over all validation frames before scoring keys."""

    def __init__(self, thresholds, prior, temperature=0.1, regularization=0.05):
        if thresholds.pitch_dependent:
            raise ValueError("AvgF loss requires three global thresholds.")
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(regularization) or regularization < 0:
            raise ValueError("AvgF temperature must be positive and regularization nonnegative.")
        self.thresholds = thresholds
        self.prior = np.asarray(prior, dtype=np.float64)
        self.temperature, self.regularization = temperature, regularization
        self.matched = np.zeros((3, 88), np.float64)
        self.predicted = np.zeros_like(self.matched)
        self.positives = np.zeros_like(self.matched)
        self.elements = np.zeros(3, np.int64)

    def add(self, probabilities, targets):
        for index, head in enumerate(HEADS):
            truth = np.asarray(targets[head], dtype=np.float64)
            probability = np.asarray(probabilities[head], dtype=np.float64)
            if probability.shape != truth.shape or truth.shape[-1] != 88:
                raise ValueError("AvgF probabilities and targets must have matching 88-key shapes.")
            scaled = (probability - getattr(self.thresholds, head)) / self.temperature
            decisions = 1 / (1 + np.exp(-np.clip(scaled, -60, 60)))
            axes = tuple(range(truth.ndim - 1))
            self.matched[index] += (decisions * truth).sum(axis=axes)
            self.predicted[index] += decisions.sum(axis=axes)
            self.positives[index] += truth.sum(axis=axes)
            self.elements[index] += truth.size // 88

    def results(self):
        if np.any(self.elements == 0):
            raise ValueError("AvgF validation needs frame, onset and offset targets.")
        losses = []
        for matched, predicted, positives, count in zip(self.matched, self.predicted, self.positives, self.elements):
            supported = positives > 0
            if supported.any():
                scores = np.stack([(1 + beta ** 2) * matched / np.maximum(predicted + beta ** 2 * positives, 1e-6)
                                   for beta in (0, 1, 2, 3)]).mean(axis=0)
                losses.append(float((1 - scores[supported]).mean()))
            if (~supported).any():
                losses.append(float((predicted[~supported] / count).mean()))
        data_loss = float(np.mean(losses))
        values = np.asarray(list(self.thresholds.summary().values()))
        prior_loss = float(self.regularization * np.mean((values - self.prior) ** 2))
        return {"avgf_loss": data_loss + prior_loss, "avgf_data_loss": data_loss, "avgf_prior_loss": prior_loss}


def avgf_accumulators(model, grid, temperature=0.1, regularization=0.05):
    if getattr(model, "architecture", None) != "onsets-fourier-recurrent":
        return {}
    prior = model.threshold_module.prior.detach().cpu().numpy()
    return {values: AvgFLoss(values, prior, temperature, regularization) for values in grid}


def event_targets(frame, notes, duration, frames_per_second=50):
    """The training dataset's two-frame event labels, on a complete timeline."""
    targets = {"frame": frame, "onset": np.zeros_like(frame, dtype=np.float32),
               "offset": np.zeros_like(frame, dtype=np.float32)}
    for note in notes:
        for head, time in (("onset", note["start"]), ("offset", note["end"])):
            if 0 <= time < duration:
                first = min(len(frame) - 1, round(time * frames_per_second))
                targets[head][first:first + 2, note["pitch"] - 21] = 1
    return targets
