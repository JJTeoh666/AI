import csv
import io
import json
import tempfile
import unittest
import wave
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.data import MaestroWindows, load_records
from piano_ml.middle_data import (EXCLUSIONS_NAME, _download_verified, _safe_path,
                                  refresh_middle_training)
from piano_ml.training import pitch_loss_weights, training_loss


def midi_bytes(pitches):
    events = b"".join(bytes([48, 0x90, pitch, 100, 48, 0x80, pitch, 0]) for pitch in pitches)
    events += bytes([0, 0xFF, 0x2F, 0])
    return (b"MThd" + (6).to_bytes(4, "big") + bytes([0, 0, 0, 1, 1, 0xE0])
            + b"MTrk" + len(events).to_bytes(4, "big") + events)


class MiddleFocusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
        self.audio = buffer.getvalue()
        rows = []
        definitions = (("old-low", "train", [29], True),
                       ("old-high", "train", [69], True),
                       ("old-focus", "train", [30, 30, 30, 68], True),
                       ("new-focus", "train", [30, 68], False),
                       ("new-outside", "train", [70], False),
                       ("validation", "validation", [50], True),
                       ("test", "test", [80], True))
        self.midi_zip, self.audio_zip = self.root / "index.zip", self.root / "audio.zip"
        with zipfile.ZipFile(self.midi_zip, "w") as midi, zipfile.ZipFile(self.audio_zip, "w") as audio:
            for name, split, pitches, local in definitions:
                wav_name, midi_name = f"{name}.wav", f"{name}.midi"
                data = midi_bytes(pitches)
                midi.writestr("maestro/" + midi_name, data)
                audio.writestr("maestro/" + wav_name, self.audio)
                if local:
                    (self.root / wav_name).write_bytes(self.audio)
                    (self.root / midi_name).write_bytes(data)
                rows.append(dict(split=split, audio_filename=wav_name, midi_filename=midi_name, duration="1"))
        with (self.root / "maestro-v3.0.0.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        self.budget_gb = (len(self.audio) + len(midi_bytes([30, 68])) + 1) / 1e9

    def tearDown(self):
        self.temporary.cleanup()

    def refresh(self, dry_run=False):
        def remote(*args, **kwargs):
            return zipfile.ZipFile(self.audio_zip)
        with patch("piano_ml.middle_data._midi_archive", return_value=self.midi_zip), \
                patch("piano_ml.middle_data.remote_zip_class", return_value=remote), redirect_stdout(io.StringIO()):
            return refresh_middle_training(self.root, self.budget_gb, dry_run=dry_run)

    def test_focus_boundaries_rare_pitch_weights_and_ordinary_fallback(self):
        data = MaestroWindows(self.root, "train", seconds=1, windows_per_file=1,
                              random_windows=True, multi_target=True, middle_sampling=0.6)
        self.assertEqual([note.pitch for _, note in data.middle_anchors], [30, 30, 30, 68])
        increments = np.diff([0, *data.middle_cumulative])
        np.testing.assert_allclose(increments, [1 / np.sqrt(3)] * 3 + [1])
        with patch("piano_ml.data.random.random", return_value=0.1):
            _, labels = data[0]
        self.assertTrue(all(30 <= note["pitch"] <= 68 for note in labels["reference_notes"]))
        with patch("piano_ml.data.random.random", return_value=0.9):
            _, labels = data[0]
        self.assertEqual(labels["reference_notes"][0]["pitch"], 29)
        no_anchors = MaestroWindows(self.root, "train", seconds=1, random_windows=True,
                                    multi_target=True, middle_sampling=1, max_files=1)
        self.assertEqual(no_anchors.middle_anchors, [])
        self.assertEqual(no_anchors[0][1]["reference_notes"][0]["pitch"], 29)
        with self.assertRaisesRegex(ValueError, "sum"):
            MaestroWindows(self.root, "train", random_windows=True, middle_sampling=0.8, bass_sampling=0.3)
        with self.assertRaisesRegex(ValueError, "only available"):
            MaestroWindows(self.root, "validation", random_windows=True, middle_sampling=0.6)
        with self.assertRaisesRegex(ValueError, "inclusive"):
            MaestroWindows(self.root, "train", random_windows=True, middle_min_note=70, middle_max_note=30)

    def test_focus_weights_both_false_positive_and_missed_notes(self):
        weights = pitch_loss_weights(edge_loss_weight=1, middle_loss_weight=2)
        self.assertTrue(torch.equal(weights[30 - 21:68 - 21 + 1], torch.full((39,), 2.0)))
        self.assertEqual(float(weights.sum()), 127)
        for truth in (0.0, 1.0):
            output = {head: torch.zeros(1, 5, 88, requires_grad=True)
                      for head in ("frame", "onset", "offset", "velocity")}
            output["pedal"] = torch.zeros(1, 5, 1, requires_grad=True)
            targets = {head: torch.full_like(logits, truth) for head, logits in output.items()}
            training_loss(output, targets, pitch_weights=weights).backward()
            for head in ("frame", "onset", "offset"):
                outside = output[head].grad[..., 69 - 21].abs().mean()
                for pitch in (30, 47, 60, 68):
                    self.assertAlmostEqual(float(output[head].grad[..., pitch - 21].abs().mean() / outside), 2)
                self.assertAlmostEqual(float(output[head].grad[..., 29 - 21].abs().mean() / outside), 1)
        with self.assertRaises(ValueError):
            pitch_loss_weights(middle_min_note=20, middle_loss_weight=2)

    def test_replacement_preserves_backups_and_heldout_splits(self):
        heldout = {name: (self.root / name).read_bytes() for name in
                   ("validation.wav", "validation.midi", "test.wav", "test.midi", "maestro-v3.0.0.csv")}
        initial = {row["audio_filename"] for row in load_records(self.root, "train")}
        plan = self.refresh(dry_run=True)
        self.assertEqual(plan["middle_range"], [30, 68])
        self.assertEqual(plan["status"], "planned")
        self.assertFalse((self.root / EXCLUSIONS_NAME).exists())
        self.assertFalse((self.root / "new-focus.wav").exists())
        self.assertEqual(len(plan["new_records"]), 1)
        self.assertEqual(plan["new_records"][0]["middle_fraction"], 1)
        result = self.refresh()
        self.assertEqual(result["status"], "complete")
        retired = {row["audio_filename"] for row in result["retired_records"]}
        active = {row["audio_filename"] for row in load_records(self.root, "train")}
        self.assertEqual(active, (initial - retired) | {"new-focus.wav"})
        self.assertTrue(all((self.root / name).read_bytes() == self.audio for name in retired))
        self.assertEqual((self.root / "new-focus.wav").read_bytes(), self.audio)
        for name, data in heldout.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        # Exclusions are applied only to the training split, even if a held-out name is present.
        manifest = self.root / EXCLUSIONS_NAME
        payload = json.loads(manifest.read_text())
        payload["excluded_audio_filenames"].extend(["validation.wav", "test.wav"])
        manifest.write_text(json.dumps(payload))
        self.assertEqual(len(load_records(self.root, "validation")), 1)
        self.assertEqual(len(load_records(self.root, "test")), 1)
        payload["excluded_audio_filenames"] = []
        manifest.write_text(json.dumps(payload))
        self.assertTrue(initial.issubset({row["audio_filename"] for row in load_records(self.root, "train")}))

    def test_failed_download_keeps_original_active_set(self):
        original = {row["audio_filename"] for row in load_records(self.root, "train")}
        downloaded = _download_verified
        calls = []
        def fail_second(remote, file, stage):
            calls.append(file["path"])
            if len(calls) == 2:
                raise OSError("interrupted transfer")
            return downloaded(remote, file, stage)
        with patch("piano_ml.middle_data._download_verified", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.refresh()
        self.assertEqual({row["audio_filename"] for row in load_records(self.root, "train")}, original)
        self.assertFalse((self.root / EXCLUSIONS_NAME).exists())
        self.assertFalse((self.root / "new-focus.wav").exists())

    def test_crc_failures_and_unsafe_paths_are_rejected(self):
        class Corrupt:
            def open(self, name):
                return io.BytesIO(b"broken")
        with patch("piano_ml.middle_data.time.sleep"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "CRC32"):
                _download_verified(Corrupt(), {"path": "bad.wav", "member": "bad.wav", "bytes": 6,
                                             "crc32": "00000000"}, self.root / "stage")
        self.assertFalse((self.root / "stage/bad.wav").exists())
        self.assertFalse((self.root / "stage/bad.wav.partial").exists())
        for unsafe in ("../outside.wav", str(self.root.parent / "outside.wav")):
            with self.assertRaises(ValueError):
                _safe_path(self.root, unsafe)


if __name__ == "__main__":
    unittest.main()
