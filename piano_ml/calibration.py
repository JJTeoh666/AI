"""Aggregate frame and note metrics for independent threshold combinations."""

import numpy as np

from .decode import DecodeCache, decode_outputs, pitch_name
from .metrics import PreparedNoteMatcher, note_f_scores, pitch_note_report, precision_recall_f1
from .model import LOW_NOTE, N_NOTES
from .thresholds import Thresholds


class Calibration:
    def __init__(self, grid: list[Thresholds], min_note_seconds=None, release_frames=2):
        if not grid:
            raise ValueError("Threshold grids cannot be empty.")
        self.grid = grid
        self.minimum = min_note_seconds
        self.release_frames = release_frames
        self.counts = {values: {key: np.zeros(3, dtype=np.int64)
                               for key in ("frame", "onset", "note")} for values in grid}
        self.pitch_counts = {values: np.zeros((N_NOTES, 3), dtype=np.int64) for values in self.counts}

    def add(self, probabilities, truth, reference, duration, window=False):
        frame_counts = {}
        cache = DecodeCache(probabilities) if reference is not None else None
        matcher = PreparedNoteMatcher(reference, duration if window else None) if reference is not None else None
        for values, counts in self.counts.items():
            if values.frame not in frame_counts:
                predicted = probabilities["frame"] >= values.frame
                frame_counts[values.frame] = [np.sum(predicted & truth), np.sum(predicted & ~truth),
                                             np.sum(~predicted & truth)]
            counts["frame"] += frame_counts[values.frame]
            if reference is None:
                continue
            notes = decode_outputs(probabilities, duration, values.frame, self.minimum,
                                   include_chords=False, onset_threshold=values.onset,
                                   offset_threshold=values.offset, release_frames=self.release_frames,
                                   _cache=cache, include_details=False)["notes"]
            for pitch, (onset_counts, values_by_pitch) in matcher.counts(notes).items():
                counts["onset"] += onset_counts
                counts["note"] += values_by_pitch
                self.pitch_counts[values][pitch - LOW_NOTE] += values_by_pitch

    def results(self, metric="note_f_avg", extra_metrics=None):
        curve = []
        pitch_reports = {}
        for values, counts in self.counts.items():
            onset = precision_recall_f1(*counts["onset"])
            note = precision_recall_f1(*counts["note"])
            report = pitch_note_report(self.pitch_counts[values], LOW_NOTE)
            for row in report["per_pitch_notes"]:
                row["name"] = pitch_name(row["pitch"])
            pitch_reports[values] = report["per_pitch_notes"]
            curve.append({"threshold": values.summary()["frame"], "thresholds": values.to_dict(),
                          **(extra_metrics or {}).get(values, {}),
                          **precision_recall_f1(*counts["frame"]), "onset_f1": onset["f1"],
                          **note_f_scores(*counts["note"]), "note_precision": note["precision"],
                          "note_recall": note["recall"],
                          **{key: value for key, value in report.items() if key != "per_pitch_notes"}})
        if metric == "avgf_loss":
            if any("avgf_loss" not in entry for entry in curve):
                raise ValueError("AvgF selection needs validation AvgF losses for each threshold set.")
            best = min(curve, key=lambda entry: entry["avgf_loss"])
        else:
            best = max(curve, key=lambda entry: (entry[metric],
                entry["onset_f1"] if metric in ("note_f1", "note_f_avg", "note_macro_f03", "note_macro_f0123") else 0, entry["f1"],
                -sum(abs(value - 0.5) for value in Thresholds(**entry["thresholds"]).summary().values())))
        return {**best, "threshold_curve": curve,
                "per_pitch_notes": pitch_reports[Thresholds(**best["thresholds"])]}
