"""Micro and per-pitch note metrics with one-to-one event matching."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def note_f_scores(tp: int, fp: int, fn: int) -> dict:
    """Note F-beta scores for beta 0..4 and their equally weighted mean.

    F0 is precision. Counts are aggregated over the validation set before
    computing these scores; an undefined score is zero.
    """
    tp, fp, fn = int(tp), int(fp), int(fn)
    scores = {}
    for beta in range(5):
        numerator = (1 + beta ** 2) * tp
        denominator = numerator + fp + beta ** 2 * fn
        scores[f"note_f{beta}"] = numerator / denominator if denominator else 0.0
    scores["note_f_avg"] = sum(scores.values()) / 5
    return scores


def note_counts(reference: list[dict], predicted: list[dict], offsets: bool = True,
                onset_tolerance: float = 0.05, offset_ratio: float = 0.2,
                offset_tolerance: float = 0.05) -> tuple[int, int, int]:
    """Total correct, extra and missed notes, using unique event matches."""
    counts = note_counts_by_pitch(reference, predicted, offsets, onset_tolerance,
                                 offset_ratio, offset_tolerance)
    return tuple(sum(values[index] for values in counts.values()) for index in range(3))


def note_counts_by_pitch(reference: list[dict], predicted: list[dict], offsets: bool = True,
                         onset_tolerance: float = 0.05, offset_ratio: float = 0.2,
                         offset_tolerance: float = 0.05) -> dict[int, tuple[int, int, int]]:
    """Maximum matching avoids counting one prediction against multiple notes.

    Require exact MIDI pitch, onset within 50 ms, and (optionally) offset
    within max(50 ms, 20% of the reference duration). Include pitches with
    only references or only predictions so their errors remain visible.
    """
    truth_by_pitch, guess_by_pitch = {}, {}
    for note in reference:
        truth_by_pitch.setdefault(int(note["pitch"]), []).append(note)
    for note in predicted:
        guess_by_pitch.setdefault(int(note["pitch"]), []).append(note)
    counts = {}
    for pitch in sorted(truth_by_pitch.keys() | guess_by_pitch.keys()):
        truth, guess = truth_by_pitch.get(pitch, []), guess_by_pitch.get(pitch, [])
        if not truth or not guess:
            counts[pitch] = (0, len(guess), len(truth))
            continue
        starts = np.array([note["start"] for note in truth])[:, None]
        ends = np.array([note["end"] for note in truth])[:, None]
        candidates = np.abs(starts - np.array([note["start"] for note in guess])) <= onset_tolerance + 1e-9
        if offsets:
            tolerance = np.maximum(offset_tolerance, offset_ratio * (ends - starts))
            candidates &= np.abs(ends - np.array([note["end"] for note in guess])) <= tolerance + 1e-9
        matching = maximum_bipartite_matching(csr_matrix(candidates), perm_type="column")
        matched = int((matching >= 0).sum())
        counts[pitch] = (matched, len(guess) - matched, len(truth) - matched)
    return counts


class PreparedNoteMatcher:
    """Prepare references once; share onset candidates between both note scores."""

    def __init__(self, reference, duration=None, margin=0.1):
        self.duration, self.margin = duration, margin
        self.cut_starts = {}
        if duration is not None:
            for note in reference:
                if note["end"] > duration - margin:
                    self.cut_starts.setdefault(int(note["pitch"]), []).append(note["start"])
            reference = interior_notes(reference, duration, margin)
        groups = {}
        for note in reference:
            groups.setdefault(int(note["pitch"]), []).append(note)
        self.truth = {}
        for pitch, notes in groups.items():
            starts = np.array([note["start"] for note in notes])[:, None]
            ends = np.array([note["end"] for note in notes])[:, None]
            self.truth[pitch] = (starts, ends, np.maximum(0.05, 0.2 * (ends - starts)))

    @staticmethod
    def matched(candidates):
        if not candidates.any():
            return 0
        if min(candidates.shape) == 1:
            return 1
        return int((maximum_bipartite_matching(csr_matrix(candidates), perm_type="column") >= 0).sum())

    def counts(self, predicted):
        guesses = {}
        for note in predicted:
            pitch = int(note["pitch"])
            if self.duration is not None:
                if not self.margin <= note["start"] < self.duration - self.margin:
                    continue
                if any(abs(note["start"] - start) <= 0.05 for start in self.cut_starts.get(pitch, ())):
                    continue
            guesses.setdefault(pitch, []).append(note)
        counts = {}
        for pitch in self.truth.keys() | guesses.keys():
            truth, guess = self.truth.get(pitch), guesses.get(pitch, [])
            if truth is None or not guess:
                values = (0, len(guess), len(truth[0]) if truth is not None else 0)
                counts[pitch] = (values, values)
                continue
            starts, ends, tolerance = truth
            candidates = np.abs(starts - np.array([note["start"] for note in guess])) <= 0.05 + 1e-9
            onset_matched = self.matched(candidates)
            candidates &= np.abs(ends - np.array([note["end"] for note in guess])) <= tolerance + 1e-9
            note_matched = self.matched(candidates)
            counts[pitch] = ((onset_matched, len(guess) - onset_matched, len(starts) - onset_matched),
                             (note_matched, len(guess) - note_matched, len(starts) - note_matched))
        return counts


def pitch_note_report(counts: np.ndarray, low_note: int = 21) -> dict:
    """Average per-key F scores after accumulating counts over all validation clips."""
    rows = []
    for column, (tp, fp, fn) in enumerate(counts):
        tp, fp, fn = int(tp), int(fp), int(fn)
        scores = note_f_scores(tp, fp, fn)
        rows.append({"pitch": low_note + column, "reference_count": tp + fn,
                     "predicted_count": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                     **precision_recall_f1(tp, fp, fn),
                     "f0": scores["note_f0"], "f2": scores["note_f2"], "f3": scores["note_f3"],
                     "f03": (scores["note_f0"] + scores["note_f3"]) / 2,
                     "f0123": sum(scores[f"note_f{beta}"] for beta in range(4)) / 4,
                     "in_macro_f1": tp + fn > 0, "in_macro_f03": tp + fn > 0,
                     "in_macro_f0123": tp + fn > 0})
    supported = [row for row in rows if row["in_macro_f1"]]
    averages = {f"note_macro_{key}": sum(row[key] for row in supported) / len(supported)
                if supported else 0.0 for key in ("f0", "f1", "f2", "f3", "f03", "f0123")}
    return {**averages,
            "note_macro_pitch_count": len(supported), "per_pitch_notes": rows}


def interior_notes(notes: list[dict], duration: float, margin: float = 0.1) -> list[dict]:
    """Exclude artificial clip-boundary events for window-based validation."""
    return [note for note in notes if note["start"] >= margin and note["end"] <= duration - margin]


def window_note_pairs(reference: list[dict], predicted: list[dict], duration: float,
                      margin: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Ignore real notes cut by the clip; retain erroneous long predictions."""
    complete = interior_notes(reference, duration, margin)
    cut = [note for note in reference if note["end"] > duration - margin]
    predicted = [note for note in predicted if margin <= note["start"] < duration - margin
                 and not any(note["pitch"] == truth["pitch"] and abs(note["start"] - truth["start"]) <= 0.05
                             for truth in cut)]
    return complete, predicted
