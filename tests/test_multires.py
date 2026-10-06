import csv
import io
import json
import random
import tempfile
import types
import unittest
import wave
import zipfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.bass_data import download_bass, select_bass_recordings
from piano_ml.data import MaestroWindows
from piano_ml.decode import decode_outputs
from piano_ml.inference import load_model, transcribe_audio
from piano_ml.model import GlobalRecurrentPianoNet, MultiResolutionPianoNet
from piano_ml.training import train, training_loss, transfer_weights
from piano_ml.__main__ import main


def midi_bytes(pitch):
    events = bytes([0, 0x90, pitch, 100, 0x83, 0x60, 0x80, pitch, 0, 0, 0xFF, 0x2F, 0])
    return b"MThd" + (6).to_bytes(4, "big") + bytes([0, 0, 0, 1, 1, 0xE0]) + b"MTrk" + len(events).to_bytes(4, "big") + events


class MultiResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        rows = []
        for split, pitch in (("train", 24), ("validation", 69), ("test", 69)):
            (self.root / f"{split}.midi").write_bytes(midi_bytes(pitch))
            with wave.open(str(self.root / f"{split}.wav"), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(np.zeros(16000, dtype="<i2").tobytes())
            rows.append({"split": split, "audio_filename": f"{split}.wav",
                         "midi_filename": f"{split}.midi", "duration": "1.0"})
        self.write_rows(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows):
        with (self.root / "maestro-v3.0.0.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def options(self, output, **changes):
        values = dict(data=str(self.root), output=str(output), architecture="onsets-multires-global", epochs=1,
                      batch_size=1, seconds=1.0, windows_per_file=2, max_files=None,
                      positive_weight=None, lr=1e-3, seed=42, workers=0, device="cpu",
                      resume=None, init_from=None, reset_optimizer=False, augment=False,
                      hidden_size=16, gru_layers=1, calibration_values=[0.4, 0.6])
        return Namespace(**{**values, **changes})

    def test_v5_transfer_preserves_predictions_and_new_branch_learns(self):
        source = GlobalRecurrentPianoNet(hidden_size=16, gru_layers=1).eval()
        model = MultiResolutionPianoNet(hidden_size=16, gru_layers=1).eval()
        self.assertTrue(transfer_weights(model, {"model": source.state_dict()}))
        audio = torch.randn(1, 16320) * 0.1
        with torch.no_grad():
            before, after = source(audio), model(audio)
        for head in before:
            self.assertTrue(torch.equal(before[head], after[head]), head)
            self.assertEqual(after[head].shape[1], 52)
        self.assertEqual(model.long_mel.spectrogram.n_fft, 8192)
        self.assertEqual(model.threshold_module.raw.numel(), 3)
        targets = {head: torch.zeros_like(logits) for head, logits in after.items()}
        targets["frame"][:, 10:30, 3] = 1
        targets["onset"][:, 10:12, 3] = 1
        targets["offset"][:, 30:32, 3] = 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            loss = training_loss(model(audio), targets, offset_loss_weight=1, release_loss_weight=0.25)
            loss.backward()
            optimizer.step()
        self.assertGreater(float(model.long_features[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.offset_refinement[0].weight.grad.abs().sum()), 0)

    def test_bass_windows_use_train_anchors_and_validation_stays_fixed(self):
        rows = [{"split": split, "audio_filename": f"{split}.wav",
                 "midi_filename": f"{split}.midi", "duration": "1.0"} for split in ("train", "validation", "test")]
        rows.append({"split": "train", "audio_filename": "treble.wav", "midi_filename": "treble.midi", "duration": "1.0"})
        (self.root / "treble.wav").write_bytes((self.root / "train.wav").read_bytes())
        (self.root / "treble.midi").write_bytes(midi_bytes(84))
        self.write_rows(rows)
        data = MaestroWindows(self.root, "train", seconds=1, windows_per_file=2,
                              random_windows=True, multi_target=True, bass_sampling=1)
        for _ in range(12):
            _, targets = data[3]  # Normally indexes the treble-only recording.
            self.assertTrue(any(note["pitch"] == 24 for note in targets["reference_notes"]))
        ordinary = MaestroWindows(self.root, "validation", seconds=1, windows_per_file=2, multi_target=True)
        first = ordinary[0][1]
        random.seed(22)
        second = ordinary[0][1]
        self.assertEqual(first["reference_notes"], second["reference_notes"])
        with self.assertRaisesRegex(ValueError, "only available"):
            MaestroWindows(self.root, "validation", random_windows=True, bass_sampling=0.5)

    def test_release_loss_pushes_activity_toward_both_sides_of_boundary(self):
        logits = {head: torch.zeros(1, 20, 88, requires_grad=True) for head in ("frame", "onset", "offset", "velocity")}
        logits["pedal"] = torch.zeros(1, 20, 1, requires_grad=True)
        truth = {head: torch.zeros_like(value) for head, value in logits.items()}
        truth["frame"][:, :10, 3] = 1
        truth["offset"][:, 10:12, 3] = 1
        baseline = training_loss(logits, truth)
        upgraded = training_loss(logits, truth, offset_loss_weight=1, release_loss_weight=0.25)
        upgraded.backward()
        self.assertGreater(float(upgraded.detach()), float(baseline.detach()))
        self.assertLess(float(logits["frame"].grad[0, 9, 3]), 0)  # Keep the key active before release.
        self.assertGreater(float(logits["frame"].grad[0, 12, 3]), 0)  # End it after release.
        self.assertLess(float(logits["offset"].grad[0, 10, 3]), 0)

    def test_release_decoder_ignores_short_dips_and_backdates_actual_silence(self):
        outputs = {head: np.zeros((50, 88), np.float32) for head in ("frame", "onset", "offset")}
        outputs["frame"][5:30, 3] = 0.9
        outputs["frame"][12:14, 3] = 0.1
        outputs["onset"][5, 3] = 0.9
        legacy = decode_outputs(outputs, 1, 0.5, include_chords=False)["notes"]
        upgraded = decode_outputs(outputs, 1, 0.5, include_chords=False, release_frames=3)["notes"]
        self.assertEqual(legacy[0]["end"], 0.24)
        self.assertEqual(upgraded[0]["end"], 0.6)
        outputs["offset"][25, 3] = 0.9
        self.assertEqual(decode_outputs(outputs, 1, 0.5, include_chords=False, release_frames=3)["notes"][0]["end"], 0.5)

    def test_training_resume_calibration_parameters_and_app_inference(self):
        checkpoint = self.root / "v6.pt"
        with redirect_stdout(io.StringIO()):
            train(self.options(checkpoint))
        latest = self.root / "v6.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["format_version"], 7)
        self.assertEqual(saved["model_config"]["architecture"], "onsets-multires-global")
        self.assertEqual(saved["training_config"]["bass_sampling"], 0.5)
        self.assertEqual(saved["training_config"]["threshold_calibration"], "validation")
        self.assertEqual(saved["training_config"]["offset_loss_weight"], 1)
        self.assertEqual(saved["optimizer"]["param_groups"][1]["params"], [])
        loaded = load_model(latest, torch.device("cpu"))
        self.assertTrue(all(abs(saved["thresholds"][head] - value) < 1e-6
                            for head, value in loaded.learned_thresholds().to_dict().items()))
        # Resume a run that already hit the stop count, explicitly renewing patience.
        saved["stale_epochs"] = 20
        torch.save(saved, latest)
        with redirect_stdout(io.StringIO()):
            train(self.options(checkpoint, resume=str(latest), hidden_size=None, gru_layers=None,
                               reset_early_stopping=True, lr=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertLessEqual(resumed["stale_epochs"], 1)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu")
        self.assertEqual(result["architecture"], "onsets-multires-global")
        self.assertNotIn("register_thresholds", result)
        self.assertEqual(len(result["thresholds"]), 3)

    def test_validation_calibration_selects_exact_notes_and_keeps_current_candidate(self):
        source = MultiResolutionPianoNet(hidden_size=16, gru_layers=1)
        path = self.root / "source.pt"
        torch.save({"model": source.state_dict(), "model_config": source.config}, path)
        seen = []
        def exact_score(model, *args, threshold_configs, **kwargs):
            self.assertIn(model.learned_thresholds(), threshold_configs)
            chosen = threshold_configs[-1]
            seen.append(chosen)
            return {"loss": 0.1, "f1": 0.6, "note_f1": 0.6, "note_f_avg": 0.6,
                    "onset_f1": 0.6, "precision": 0.6, "recall": 0.6,
                    "threshold": chosen.frame, "thresholds": chosen.to_dict(), "threshold_curve": []}
        with patch("piano_ml.training.score", side_effect=exact_score), redirect_stdout(io.StringIO()):
            train(self.options(self.root / "calibration.pt", init_from=str(path)))
        saved = torch.load(self.root / "calibration.last.pt", weights_only=True)
        loaded = load_model(self.root / "calibration.last.pt", torch.device("cpu"))
        self.assertTrue(all(abs(loaded.learned_thresholds().to_dict()[head] - value) < 1e-6
                            for head, value in seen[-1].to_dict().items()))
        self.assertEqual(saved["thresholds"], seen[-1].to_dict())
        self.assertFalse(loaded.learned_thresholds().pitch_dependent)

    def test_bass_selection_respects_budget_and_favors_scarce_pitches(self):
        candidates = [{"audio_filename": "common", "bytes": 10, "bass_count": 10, "pitch_counts": {36: 10}},
                      {"audio_filename": "rare", "bytes": 10, "bass_count": 10, "pitch_counts": {24: 10}},
                      {"audio_filename": "large", "bytes": 100, "bass_count": 100, "pitch_counts": {21: 100}}]
        selected, coverage = select_bass_recordings(candidates, {36: 1000}, 10)
        self.assertEqual([row["audio_filename"] for row in selected], ["rare"])
        self.assertEqual(coverage[24], 10)

    def test_bass_download_only_adds_verified_official_train_pairs(self):
        with (self.root / "maestro-v3.0.0.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for name, split in (("bass", "train"), ("validation-bass", "validation"), ("test-bass", "test")):
            rows.append({"split": split, "audio_filename": name + ".wav", "midi_filename": name + ".midi", "duration": "1.0"})
        self.write_rows(rows)
        midi_path, wav_path = self.root / "midi.zip", self.root / "audio.zip"
        wav_bytes = (self.root / "train.wav").read_bytes()
        with zipfile.ZipFile(midi_path, "w") as midi, zipfile.ZipFile(wav_path, "w") as audio:
            for row in rows:
                midi.writestr(row["midi_filename"], midi_bytes(24 if "bass" in row["audio_filename"] else 69))
                audio.writestr(row["audio_filename"], wav_bytes)
        remote = lambda *args, **kwargs: zipfile.ZipFile(wav_path)
        with patch("piano_ml.bass_data._midi_archive", return_value=midi_path), \
             patch("piano_ml.bass_data.remote_zip_class", return_value=remote), redirect_stdout(io.StringIO()):
            report = download_bass(self.root, gigabytes=0.0001)
        self.assertEqual(report["added_recordings"], 1)
        self.assertEqual(report["status"], "complete")
        self.assertEqual((self.root / "bass.wav").read_bytes(), wav_bytes)
        self.assertFalse((self.root / "validation-bass.wav").exists())
        self.assertFalse((self.root / "test-bass.wav").exists())
        self.assertLessEqual(report["downloaded_additional_bytes"], report["budget_bytes"])

    def test_cli_auto_and_version_six_options(self):
        with patch("sys.argv", ["piano_ml", "train", "--long-fft", "8192", "--bass-sampling", "0.5"]), \
             patch("piano_ml.__main__.train") as run:
            main()
        options = run.call_args.args[0]
        self.assertEqual(options.architecture, "auto")
        self.assertEqual(options.patience, 20)
        self.assertEqual(options.long_fft, 8192)

    def test_periodic_calibration_validates_every_epoch_and_forces_search_before_stop(self):
        phases = []
        scores = iter([0.6, 0.5, 0.65, 0.5, 0.5])
        def fake_score(model, *args, threshold_configs, **kwargs):
            phases.append(len(threshold_configs) > 1)
            value = next(scores)
            chosen = threshold_configs[0]
            return {"loss": 0.1, "f1": value, "note_f1": value, "note_f_avg": value,
                    "onset_f1": value, "precision": value, "recall": value,
                    "threshold": chosen.frame, "thresholds": chosen.to_dict(), "threshold_curve": []}
        checkpoint = self.root / "periodic.pt"
        with patch("piano_ml.training.score", side_effect=fake_score), redirect_stdout(io.StringIO()):
            train(self.options(checkpoint, epochs=5, patience=2, calibration_every=99))
        self.assertEqual(phases, [True, False, True, False, True])
        best = torch.load(checkpoint, weights_only=True)
        latest = torch.load(self.root / "periodic.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 3)
        self.assertEqual(latest["stale_epochs"], 2)
        self.assertEqual(latest["training_config"]["calibration_every"], 99)
        rows = [json.loads(line) for line in checkpoint.with_suffix(".history.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1]["calibration_search"], "current")
        self.assertEqual(rows[2]["calibration_search"], "full")
        self.assertTrue(all(row["train_seconds"] >= 0 and row["validation_seconds"] >= 0 for row in rows))

    def test_calibration_schedule_survives_resume_without_resetting_best_or_optimizer(self):
        def fake_score(model, *args, threshold_configs, **kwargs):
            chosen = threshold_configs[0]
            return {"loss": 0.1, "f1": 0.5, "note_f1": 0.5, "note_f_avg": 0.5,
                    "onset_f1": 0.5, "precision": 0.5, "recall": 0.5,
                    "threshold": chosen.frame, "thresholds": chosen.to_dict(), "threshold_curve": []}
        checkpoint = self.root / "resume-periodic.pt"
        with patch("piano_ml.training.score", side_effect=fake_score), redirect_stdout(io.StringIO()):
            train(self.options(checkpoint, calibration_every=5))
        latest = self.root / "resume-periodic.last.pt"
        saved = torch.load(latest, weights_only=True)
        saved.update(epoch=3, best_score=0.7, stale_epochs=2)
        # Mimic a pre-optimization v6 checkpoint without the new interval field.
        saved["training_config"].pop("calibration_every")
        torch.save(saved, latest)
        phases = []
        def resumed_score(model, *args, threshold_configs, **kwargs):
            phases.append(len(threshold_configs) > 1)
            return fake_score(model, threshold_configs=threshold_configs)
        console = io.StringIO()
        with patch("piano_ml.training.score", side_effect=resumed_score), redirect_stdout(console):
            train(self.options(checkpoint, epochs=3, resume=str(latest), hidden_size=None, gru_layers=None, lr=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(phases, [False, True, True])  # Epoch 4, scheduled epoch 5, final epoch 6.
        self.assertEqual(resumed["best_score"], 0.7)
        self.assertEqual(resumed["stale_epochs"], 5)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertIn("optimizer=restored", console.getvalue())
        self.assertEqual(resumed["training_config"]["calibration_every"], 5)

    def test_every_one_retains_exhaustive_calibration_each_epoch(self):
        counts = []
        def fake_score(model, *args, threshold_configs, **kwargs):
            counts.append(len(threshold_configs))
            chosen = threshold_configs[0]
            return {"loss": 0.1, "f1": 0.5, "note_f1": 0.5, "note_f_avg": 0.5,
                    "onset_f1": 0.5, "precision": 0.5, "recall": 0.5,
                    "threshold": chosen.frame, "thresholds": chosen.to_dict(), "threshold_curve": []}
        with patch("piano_ml.training.score", side_effect=fake_score), redirect_stdout(io.StringIO()):
            train(self.options(self.root / "every.pt", epochs=3, calibration_every=1))
        self.assertTrue(all(count > 1 for count in counts))


if __name__ == "__main__":
    unittest.main()
