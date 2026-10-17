"""Atomic checkpoint replacement under temporary Windows file locks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from piano_ml.training import save_checkpoint


class CheckpointIOTest(unittest.TestCase):
    @staticmethod
    def locked():
        error = PermissionError("checkpoint temporarily locked")
        error.winerror = 32
        return error

    def test_transient_lock_retries_atomic_replace_without_rewriting_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint({"epoch": 1}, path)
            original, calls = Path.replace, []

            def replace(source, target):
                calls.append((source, target))
                if len(calls) == 1:
                    self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
                    raise self.locked()
                return original(source, target)

            with patch.object(Path, "replace", replace), patch("piano_ml.training.time.sleep") as sleep:
                save_checkpoint({"epoch": 2}, path)
            self.assertEqual(len(calls), 2)
            sleep.assert_called_once_with(.05)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 2)
            self.assertFalse(path.with_name("model.pt.partial").exists())

    def test_persistent_lock_has_bounded_retries_and_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint({"epoch": 1}, path)
            with patch.object(Path, "replace", side_effect=self.locked()) as replace, \
                    patch("piano_ml.training.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    save_checkpoint({"epoch": 2}, path)
            self.assertEqual(replace.call_count, 5)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [.05, .1, .2, .4])
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)

    def test_other_permission_errors_fail_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "replace", side_effect=PermissionError("permission denied")), \
                    patch("piano_ml.training.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    save_checkpoint({"epoch": 1}, Path(directory) / "model.pt")
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
