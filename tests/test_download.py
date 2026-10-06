"""Check that additional training data cannot enter held-out splits."""

import csv
import io
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from piano_ml.download import download_additional


class AdditionalTrainingDownloadTests(unittest.TestCase):
    def test_training_only_skips_existing_pairs_and_preserves_held_out_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split, names in (("train", ("existing", "new", "incomplete")),
                                 ("validation", ("held-validation",)),
                                 ("test", ("held-test",))):
                for name in names:
                    rows.append(dict(split=split, audio_filename=f"{name}.wav",
                                     midi_filename=f"{name}.midi", duration="1",
                                     canonical_composer="Composer", canonical_title=name))
            with (root / "maestro-v3.0.0.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            archives = {}
            for kind, field in (("audio", "audio_filename"), ("midi", "midi_filename")):
                path = root / f"{kind}.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    for row in rows:
                        archive.writestr(row[field], b"audio" * 200 if kind == "audio" else b"midi" * 10)
                archives[kind] = path
            (root / "existing.wav").write_bytes(b"audio" * 200)
            (root / "existing.midi").write_bytes(b"midi" * 10)
            (root / "incomplete.wav").write_bytes(b"truncated")
            (root / "held-validation.wav").write_bytes(b"keep validation")
            (root / "held-test.midi").write_bytes(b"keep test")

            def remote_zip(url, **kwargs):
                return zipfile.ZipFile(archives["midi" if "-midi.zip" in url else "audio"])

            with patch("piano_ml.download.remote_zip_class", return_value=remote_zip), redirect_stdout(io.StringIO()):
                plan = download_additional(root, gigabytes=0.00002, train_only=True, dry_run=True)
                self.assertEqual(plan["split_counts"], dict(train=2, validation=0, test=0))
                self.assertLessEqual(plan["planned_additional_bytes"], 20_000)
                self.assertFalse((root / "new.wav").exists())
                report = download_additional(root, gigabytes=0.00002, train_only=True)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["downloaded_additional_bytes"], 2080)
                self.assertEqual((root / "incomplete.wav").read_bytes(), b"audio" * 200)
                self.assertEqual((root / "new.midi").read_bytes(), b"midi" * 10)
                self.assertTrue(all(row["split"] == "train" for row in report["records"]))
                self.assertEqual((root / "held-validation.wav").read_bytes(), b"keep validation")
                self.assertEqual((root / "held-test.midi").read_bytes(), b"keep test")
                repeated = download_additional(root, gigabytes=0.00002, train_only=True)
                self.assertEqual(repeated["downloaded_additional_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
