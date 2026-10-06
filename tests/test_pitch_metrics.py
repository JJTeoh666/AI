import unittest
from unittest.mock import patch

import numpy as np

from piano_ml.calibration import Calibration
from piano_ml.metrics import note_counts, note_counts_by_pitch, pitch_note_report
from piano_ml.thresholds import Thresholds


def note(pitch, start=0.2, end=0.6):
    return {"pitch": pitch, "start": start, "end": end}


class PitchMetricsTest(unittest.TestCase):
    def test_unique_matches_offset_errors_and_unrepresented_pitches(self):
        reference = [note(24), note(24, 0.22), note(26), note(28)]
        predicted = [note(24, 0.21), note(26, end=0.9), note(33)]
        counts = note_counts_by_pitch(reference, predicted)
        self.assertEqual(counts, {24: (1, 0, 1), 26: (0, 1, 1), 28: (0, 0, 1), 33: (0, 1, 0)})
        self.assertEqual(note_counts(reference, predicted), (1, 2, 3))
        self.assertEqual(note_counts_by_pitch(reference, predicted, offsets=False)[26], (1, 0, 0))
        self.assertEqual(note_counts([], []), (0, 0, 0))

    def test_macro_exposes_a_rare_missed_pitch_and_excludes_unsupported_keys(self):
        counts = np.zeros((88, 3), np.int64)
        counts[60 - 21] = (100, 0, 0)
        counts[24 - 21] = (0, 0, 1)
        counts[33 - 21] = (0, 8, 0)
        report = pitch_note_report(counts)
        self.assertEqual(report["note_macro_pitch_count"], 2)
        self.assertEqual(report["note_macro_f1"], 0.5)
        self.assertEqual(report["per_pitch_notes"][33 - 21]["fp"], 8)
        self.assertFalse(report["per_pitch_notes"][33 - 21]["in_macro_f1"])
        self.assertEqual(report["per_pitch_notes"][24 - 21]["reference_count"], 1)
        empty = pitch_note_report(np.zeros((88, 3), np.int64))
        self.assertEqual(empty["note_macro_f1"], 0)
        self.assertEqual(empty["note_macro_pitch_count"], 0)

    def test_counts_accumulate_across_windows_before_pitch_scores(self):
        calibration = Calibration([Thresholds()])
        probabilities = {head: np.zeros((50, 88), np.float32) for head in ("frame", "onset", "offset")}
        references = [[note(24), note(24, 0.7, 0.9), note(60)], [note(24), note(60), note(60, 0.7, 0.9)]]
        guesses = [[note(24), note(60), note(33)], references[1]]
        with patch("piano_ml.calibration.decode_outputs", side_effect=[{"notes": notes} for notes in guesses]):
            for reference in references:
                calibration.add(probabilities, np.zeros((50, 88), bool), reference, 1)
        report = calibration.results()
        rows = {row["name"]: row for row in report["per_pitch_notes"]}
        self.assertEqual(len(rows), 88)
        self.assertEqual((rows["C1"]["tp"], rows["C1"]["fn"], rows["C1"]["reference_count"]), (2, 1, 3))
        self.assertAlmostEqual(rows["C1"]["f1"], 0.8)
        self.assertAlmostEqual(report["note_macro_f1"], 0.9)
        self.assertEqual(report["note_macro_pitch_count"], 2)
        self.assertAlmostEqual(report["note_precision"], 5 / 6)
        self.assertAlmostEqual(report["note_recall"], 5 / 6)
        self.assertEqual(rows["A1"]["reference_count"], 0)
        self.assertEqual(rows["A1"]["predicted_count"], 1)
        self.assertNotIn("per_pitch_notes", report["threshold_curve"][0])

    def test_window_boundaries_are_excluded_consistently_from_pitch_metrics(self):
        calibration = Calibration([Thresholds()])
        references = [note(24, -0.2, 0.4), note(26, 0.3, 1.2), note(28)]
        guesses = [note(24, 0.01, 0.4), note(26, 0.3, 1.0), note(28), note(29, 0.3, 1.0)]
        probabilities = {head: np.zeros((50, 88), np.float32) for head in ("frame", "onset", "offset")}
        with patch("piano_ml.calibration.decode_outputs", return_value={"notes": guesses}):
            calibration.add(probabilities, np.zeros((50, 88), bool), references, 1, window=True)
        report = calibration.results()
        rows = {row["pitch"]: row for row in report["per_pitch_notes"]}
        self.assertEqual(rows[24]["reference_count"], 0)
        self.assertEqual(rows[26]["reference_count"], 0)
        self.assertEqual(rows[28]["tp"], 1)
        self.assertEqual(rows[29]["fp"], 1)
        self.assertEqual(report["note_macro_pitch_count"], 1)
        self.assertEqual(report["note_macro_f1"], 1)
        self.assertAlmostEqual(report["note_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
