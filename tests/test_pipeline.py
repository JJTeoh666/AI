import csv
import json
import tempfile
import unittest
import wave
import threading
import zipfile
import types
import io
from contextlib import redirect_stdout
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.__main__ import main as cli_main, predict, train
from piano_ml.data import MaestroWindows
from piano_ml.decode import chord_name, decode
from piano_ml.midi import read_notes, read_performance
from piano_ml.inference import transcribe_audio, load_model
from piano_ml.model import CalibratedPianoNet, OnsetsPianoNet, PianoNet, RecurrentPianoNet, RecurrentConvBlock, GlobalRecurrentPianoNet
from piano_ml.download import download_validation
from piano_ml.learned_thresholds import GlobalThresholds, RegisterThresholds
from piano_ml.training import transfer_weights
from piano_ml.metrics import note_counts, note_f_scores, window_note_pairs
from piano_ml.decode import decode_outputs
from piano_ml.evaluation import evaluate
from piano_ml.thresholds import Thresholds, saved_thresholds, threshold_grid
from piano_ml.calibration import Calibration
from piano_ml.viewer import active_at, draw_prediction, read_prediction
from piano_ml.score import draw_staff, note_value, staff_pitch
from piano_ml.synthesis import render_result_wav, RELEASE_SECONDS
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def midi_bytes(pitch=69):
    # Type-0 MIDI: one quarter-note at 120 BPM.
    events = bytes([0, 0x90, pitch, 100, 0x83, 0x60, 0x80, pitch, 0,
                    0, 0xFF, 0x2F, 0])
    return b"MThd" + (6).to_bytes(4, "big") + bytes([0, 0, 0, 1, 1, 0xE0]) + \
        b"MTrk" + len(events).to_bytes(4, "big") + events


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        rows = []
        samples = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 32767).astype("<i2")
        for split in ("train", "validation", "test"):
            midi = f"{split}.midi"
            audio = f"{split}.wav"
            (self.root / midi).write_bytes(midi_bytes())
            with wave.open(str(self.root / audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples.tobytes())
            rows.append({"split": split, "midi_filename": midi,
                         "audio_filename": audio, "duration": "1.0"})
        with (self.root / "maestro-v3.0.0.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_midi_alignment_and_chord(self):
        notes = read_notes(self.root / "train.midi")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitch, 69)
        self.assertAlmostEqual(notes[0].end, 0.5)
        _, labels = MaestroWindows(self.root, "train", seconds=1.0,
                                   windows_per_file=1)[0]
        self.assertEqual(labels.shape, (51, 88))
        self.assertEqual(labels[0, 69 - 21], 1)
        self.assertEqual(labels[26, 69 - 21], 0)
        self.assertEqual(chord_name([60, 64, 67, 72]), "C major")
        probabilities = np.zeros((50, 88), np.float32)
        for pitch in (60, 64, 67):
            probabilities[:, pitch - 21] = 0.9
        result = decode(probabilities, 1.0)
        self.assertEqual(len(result["notes"]), 3)
        self.assertEqual(result["chords"][0]["name"], "C major")

    def test_train_checkpoint_and_inference(self):
        checkpoint = self.root / "model.pt"
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=1,
                        batch_size=1, seconds=1.0, windows_per_file=1,
                        max_files=None, positive_weight=5.0, lr=1e-3,
                        seed=42, workers=0, device="cpu", resume=None,
                        reset_optimizer=False)
        train(options)
        self.assertTrue(checkpoint.exists())
        latest = self.root / "model.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["epoch"], 1)
        self.assertIn("optimizer", saved)
        parts = saved["training_metrics"]["loss_components"]
        self.assertAlmostEqual(parts["frame"], saved["training_metrics"]["total_loss"], places=6)
        self.assertTrue(all(value == 0 for name, value in parts.items() if name != "frame"))
        self.assertIn("frame_bce", saved["validation"])
        initial_step = next(iter(saved["optimizer"]["state"].values()))["step"].item()
        options.resume = str(latest)
        options.lr = 3e-4
        train(options)
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        resumed_step = next(iter(resumed["optimizer"]["state"].values()))["step"].item()
        self.assertEqual(resumed_step, initial_step + 1)
        self.assertEqual(resumed["optimizer"]["param_groups"][0]["lr"], 3e-4)
        output = self.root / "prediction.json"
        predict(Namespace(audio=str(self.root / "test.wav"),
                          checkpoint=str(checkpoint), output=str(output),
                          threshold=0.5, device="cpu"))
        result = json.loads(output.read_text())
        self.assertIn("notes", result)
        self.assertIn("chords", result)
        self.assertEqual(result["duration"], 1.0)

    def test_resume_legacy_checkpoint(self):
        legacy = self.root / "legacy.pt"
        from piano_ml.model import PianoNet
        torch.save({"model": PianoNet().state_dict(), "epoch": 7}, legacy)
        output = self.root / "continued.pt"
        train(Namespace(data=str(self.root), output=str(output), epochs=1,
                        batch_size=1, seconds=1.0, windows_per_file=1,
                        max_files=None, positive_weight=5.0, lr=1e-3,
                        seed=42, workers=0, device="cpu", resume=str(legacy),
                        reset_optimizer=False))
        saved = torch.load(self.root / "continued.last.pt", weights_only=True)
        self.assertEqual(saved["epoch"], 8)
        self.assertIn("optimizer", saved)

    def test_chunk_boundaries_do_not_duplicate_frames(self):
        audio = self.root / "long.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(128640, dtype="<i2").tobytes())  # 8.04 seconds

        class FakeModel:
            def __call__(self, samples):
                return torch.zeros((1, 201, 88))

        progress = []
        with patch("piano_ml.inference.load_model", return_value=FakeModel()), \
                patch("piano_ml.inference.decode", return_value={"notes": [], "chords": []}) as decoder:
            result = transcribe_audio(audio, "fake.pt", device="cpu",
                                      progress=lambda done, total: progress.append((done, total)))
            self.assertEqual(decoder.call_args.args[0].shape, (402, 88))
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(result["duration"], 8.04)

    def test_viewer_json_and_plot(self):
        result = {"duration": 1.0, "notes": [
            {"pitch": pitch, "start": 0.1, "end": 0.8, "confidence": 0.9}
            for pitch in (60, 64, 67)],
            "chords": [{"name": "C major", "start": 0.1, "end": 0.8}]}
        path = self.root / "result.json"
        path.write_text(json.dumps(result))
        result = read_prediction(path)
        self.assertEqual(active_at(result, 0.5), (["C4", "E4", "G4"], ["C major"]))
        self.assertEqual(active_at(result, 0.9), ([], []))
        figure = Figure(figsize=(9, 5))
        canvas = FigureCanvasAgg(figure)
        axes = draw_prediction(figure, result)
        canvas.draw()
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[2].get_xlim(), (0.0, 1.0))

    def test_onset_decoder_repeated_notes_and_short_notes(self):
        probabilities = {key: np.zeros((50, 88), np.float32)
                         for key in ("frame", "onset", "offset", "velocity")}
        column = 60 - 21
        probabilities["frame"][5:40, column] = 0.9
        probabilities["onset"][[5, 25], column] = 0.95
        probabilities["offset"][[20, 40], column] = 0.95
        probabilities["velocity"][[5, 25], column] = [0.5, 0.8]
        # A separate 40-ms note should survive the v2 decoder.
        probabilities["frame"][30:32, 64 - 21] = 0.9
        probabilities["onset"][30, 64 - 21] = 0.95
        probabilities["offset"][32, 64 - 21] = 0.95
        probabilities["pedal"] = np.zeros((50, 1), np.float32)
        probabilities["pedal"][10:45] = 0.9
        result = decode_outputs(probabilities, 1.0, 0.5)
        notes = [note for note in result["notes"] if note["pitch"] == 60]
        self.assertEqual([(note["start"], note["end"]) for note in notes], [(0.1, 0.4), (0.5, 0.8)])
        self.assertEqual([note["velocity"] for note in notes], [64, 102])
        self.assertTrue(any(note["pitch"] == 64 for note in result["notes"]))
        self.assertEqual(result["pedals"], [{"start": 0.2, "end": 0.9}])

    @staticmethod
    def independent_probabilities(frames=50):
        probabilities = {key: np.zeros((frames, 88), np.float32)
                         for key in ("frame", "onset", "offset")}
        column = 60 - 21
        probabilities["frame"][5:40, column] = 0.8
        probabilities["onset"][5, column] = 0.4
        probabilities["offset"][20, column] = 0.35
        probabilities["offset"][35, column] = 0.9
        return probabilities

    def test_each_decoder_threshold_controls_its_own_head(self):
        probabilities = self.independent_probabilities()
        options = dict(frame_threshold=0.7, onset_threshold=0.3, offset_threshold=0.5)
        result = decode_outputs(probabilities, 1, 0.5, **options)
        self.assertEqual([(note["start"], note["end"]) for note in result["notes"]], [(0.1, 0.7)])
        options["offset_threshold"] = 0.3
        self.assertEqual(decode_outputs(probabilities, 1, 0.5, **options)["notes"][0]["end"], 0.4)
        options["onset_threshold"] = 0.45
        self.assertEqual(decode_outputs(probabilities, 1, 0.5, **options)["notes"], [])
        options.update(onset_threshold=0.3, frame_threshold=0.9)
        self.assertEqual(decode_outputs(probabilities, 1, 0.5, **options)["notes"], [])
        with self.assertRaises(ValueError):
            decode_outputs(probabilities, 1, 0.5, offset_threshold=float("nan"))

    def test_independent_grid_selects_the_correct_combination(self):
        probabilities = self.independent_probabilities()
        truth = np.zeros((50, 88), bool)
        truth[5:20, 39] = True
        reference = [{"pitch": 60, "start": 0.1, "end": 0.4}]
        grid = threshold_grid(Thresholds(), [0.7, 0.9], [0.3, 0.5], [0.3, 0.5])
        calibration = Calibration(grid)
        calibration.add(probabilities, truth, reference, 1.0)
        result = calibration.results("note_f1")
        self.assertEqual(len(result["threshold_curve"]), 8)
        self.assertEqual(result["thresholds"], {"frame": 0.7, "onset": 0.3, "offset": 0.3})
        self.assertEqual(result["note_f1"], 1.0)
        self.assertEqual(saved_thresholds({"threshold": 0.65}), Thresholds(0.65, 0.65, 0.5))
        with self.assertRaises(ValueError):
            threshold_grid(Thresholds(), frame_values=[0.5], shared_values=[0.5])

    def test_checkpoint_thresholds_and_independent_inference_override(self):
        model = OnsetsPianoNet(hidden_size=16, gru_layers=1)
        checkpoint = self.root / "separate.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config, "threshold": 0.99,
                    "thresholds": {"frame": 0.7, "onset": 0.3, "offset": 0.3}}, checkpoint)
        loaded = load_model(checkpoint, torch.device("cpu"))
        self.assertEqual(loaded.detection_threshold, 0.7)

        def forward(waves):
            frames = waves.shape[1] // 320 + 1
            return {key: torch.logit(torch.from_numpy(value).clamp(0.001, 0.999)).unsqueeze(0)
                    for key, value in self.independent_probabilities(frames).items()}

        with patch("piano_ml.inference.load_model", return_value=loaded), patch.object(loaded, "forward", side_effect=forward):
            result = transcribe_audio(self.root / "test.wav", checkpoint, device="cpu")
            self.assertEqual(result["notes"][0]["end"], 0.4)
            result = transcribe_audio(self.root / "test.wav", checkpoint, device="cpu", offset_threshold=0.5)
            self.assertEqual(result["notes"][0]["end"], 0.7)
            self.assertEqual(result["thresholds"], {"frame": 0.7, "onset": 0.3, "offset": 0.5})

    def test_training_saves_and_resumes_independent_threshold_grid(self):
        checkpoint = self.root / "independent.pt"
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=1,
                            batch_size=1, seconds=1.0, windows_per_file=1,
                            max_files=None, positive_weight=5.0, lr=1e-3,
                            seed=42, workers=0, device="cpu", resume=None,
                            reset_optimizer=False, architecture="onsets", hidden_size=16,
                            gru_layers=1, thresholds=None, frame_thresholds=[0.65],
                            onset_thresholds=[0.25], offset_thresholds=[0.4], augment=False)
        train(options)
        latest = self.root / "independent.last.pt"
        saved = torch.load(latest, weights_only=True)
        expected = {"frame": 0.65, "onset": 0.25, "offset": 0.4}
        self.assertEqual(saved["thresholds"], expected)
        self.assertEqual(saved["training_config"]["threshold_grid"], [expected])
        options.resume = str(latest)
        options.frame_thresholds = options.onset_thresholds = options.offset_thresholds = None
        train(options)
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["thresholds"], expected)

    def test_learned_threshold_gradient_improves_decisions_without_changing_classifier(self):
        module = RegisterThresholds()
        probabilities = torch.full((1, 200, 88), 0.01)
        probabilities[:, :100, 48] = 0.8
        probabilities[:, 100:, 48] = 0.6
        logits = torch.logit(probabilities).requires_grad_()
        truth = torch.zeros_like(probabilities)
        truth[:, :100, 48] = 1
        outputs = {head: logits for head in ("frame", "onset", "offset")}
        targets = {head: truth for head in outputs}
        optimizer = torch.optim.Adam(module.parameters(), lr=0.05)
        original = module.values().detach().clone()
        initial_loss = float(module.loss(outputs, targets, temperature=0.03).detach())
        for _ in range(40):
            optimizer.zero_grad()
            loss = module.loss(outputs, targets, temperature=0.03)
            loss.backward()
            optimizer.step()
        self.assertIsNone(logits.grad)
        self.assertLess(float(loss.detach()), initial_loss)
        self.assertGreater(float(module.values()[0, 1].detach()), 0.6)
        self.assertTrue(torch.equal(original[:, [0, 2]], module.values().detach()[:, [0, 2]]))
        self.assertTrue(torch.all((module.values() > 0.05) & (module.values() < 0.95)))

    def test_register_thresholds_use_correct_boundary_pitches_and_preserve_overrides(self):
        onset_values = (0.5,) * 27 + (0.7,) * 36 + (0.5,) * 25
        offset_values = (0.5,) * 27 + (0.5,) * 36 + (0.9,) * 25
        thresholds = Thresholds(frame=0.5, onset=onset_values, offset=offset_values)
        probabilities = {head: np.zeros((50, 88), np.float32) for head in ("frame", "onset", "offset")}
        for pitch in (21, 47, 48, 83, 84, 108):
            column = pitch - 21
            probabilities["frame"][5:40, column] = 0.8
            probabilities["onset"][5, column] = 0.6
            probabilities["offset"][20, column] = 0.7
        decoded = decode_outputs(probabilities, 1, thresholds.frame,
                                 onset_threshold=thresholds.onset, offset_threshold=thresholds.offset)
        self.assertEqual([note["pitch"] for note in decoded["notes"]], [21, 47, 84, 108])
        self.assertEqual([note["end"] for note in decoded["notes"]], [0.4, 0.4, 0.8, 0.8])
        calibration = Calibration([thresholds])
        truth = probabilities["frame"] >= 0.5
        calibration.add(probabilities, truth, decoded["notes"], 1)
        self.assertEqual(calibration.results()["note_f1"], 1)
        override = thresholds.override(offset=0.65)
        self.assertEqual(override.onset, thresholds.onset)
        self.assertEqual(override.offset, 0.65)
        self.assertEqual(saved_thresholds({"thresholds": thresholds.to_dict()}), thresholds)
        with self.assertRaises(ValueError):
            Thresholds(frame=[0.5] * 87)

    def test_calibrated_model_transfer_training_resume_and_model_parameter_inference(self):
        original = OnsetsPianoNet(hidden_size=16, gru_layers=1)
        source = self.root / "v2.pt"
        torch.save({"model": original.state_dict(), "model_config": original.config,
                    "thresholds": {"frame": 0.5, "onset": 0.5, "offset": 0.7}}, source)
        upgraded = CalibratedPianoNet(hidden_size=16, gru_layers=1)
        self.assertTrue(transfer_weights(upgraded, torch.load(source, weights_only=True)))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, upgraded.state_dict()[key]), key)
        checkpoint = self.root / "v3.pt"
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=1,
                            batch_size=1, seconds=1.0, windows_per_file=1,
                            max_files=None, positive_weight=5.0, lr=1e-3,
                            seed=42, workers=0, device="cpu", resume=None,
                            reset_optimizer=False, architecture="onsets-calibrated", init_from=str(source),
                            hidden_size=None, gru_layers=None, thresholds_only=True, augment=False)
        train(options)
        latest = self.root / "v3.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["format_version"], 5)
        self.assertEqual(saved["model_config"]["hidden_size"], 16)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, saved["model"][key]), key)
        self.assertFalse(torch.equal(upgraded.threshold_module.raw, saved["model"]["threshold_module.raw"]))
        options.resume, options.init_from = str(latest), None
        train(options)
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["optimizer"]["param_groups"][1]["lr"], 0.01)
        # Loading uses learned tensors, even if scalar metadata is stale.
        resumed["thresholds"] = {"frame": 0.9, "onset": 0.9, "offset": 0.9}
        torch.save(resumed, latest)
        loaded = load_model(latest, torch.device("cpu"))
        expected = loaded.learned_thresholds()
        self.assertEqual(loaded.decoding_thresholds, expected)
        prediction = transcribe_audio(self.root / "test.wav", latest, device="cpu", offset_threshold=0.6)
        self.assertEqual(prediction["architecture"], "onsets-calibrated")
        self.assertEqual(prediction["thresholds"]["frame"], list(expected.frame))
        self.assertEqual(prediction["thresholds"]["offset"], 0.6)
        self.assertEqual(len(prediction["register_thresholds"]), 3)
        options.thresholds_only = False
        train(options)
        joint = torch.load(latest, weights_only=True)
        self.assertEqual(joint["epoch"], 3)
        self.assertTrue(joint["optimizer"]["param_groups"][0]["params"])
        self.assertFalse(torch.equal(original.onset_head.weight, joint["model"]["onset_head.weight"]))

    def test_learned_model_keeps_starting_checkpoint_when_note_accuracy_falls(self):
        original = OnsetsPianoNet(hidden_size=16, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": original.state_dict(), "model_config": original.config}, source)
        output = self.root / "guarded.pt"
        options = Namespace(data=str(self.root), output=str(output), epochs=2,
                            batch_size=1, seconds=1.0, windows_per_file=1,
                            max_files=None, positive_weight=5.0, lr=1e-3,
                            seed=42, workers=0, device="cpu", resume=None,
                            reset_optimizer=False, architecture="onsets-calibrated", init_from=str(source),
                            thresholds_only=True, augment=False)
        scores = iter([0.6, 0.5, 0.4])
        def decreasing_score(model, *args, **kwargs):
            value = next(scores)
            thresholds = model.learned_thresholds()
            return {"loss": 0.1, "frame_bce": 0.1, "f1": value, "onset_f1": value, "note_f1": value, "note_f_avg": value,
                    "precision": value, "recall": value, "threshold": thresholds.summary()["frame"],
                    "thresholds": thresholds.to_dict(), "threshold_curve": []}
        with patch("piano_ml.training.score", side_effect=decreasing_score):
            train(options)
        best = torch.load(output, weights_only=True)
        latest = torch.load(self.root / "guarded.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 0)
        self.assertEqual(best["best_score"], 0.6)
        self.assertEqual(latest["epoch"], 2)
        self.assertEqual(latest["stale_epochs"], 2)
        self.assertFalse(torch.equal(best["model"]["threshold_module.raw"], latest["model"]["threshold_module.raw"]))

    def test_evaluation_saves_calibration_in_model_parameters(self):
        model = CalibratedPianoNet(hidden_size=16, gru_layers=1)
        source = self.root / "learned.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        output = self.root / "calibrated.pt"
        options = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                            threshold=None, thresholds=None, frame_threshold=0.6, selection_metric="note_f1",
                            min_note_seconds=None, full_recordings=True, max_files=1,
                            calibrate_output=str(output), output=None, device="cpu")
        result = evaluate(options)
        loaded = load_model(output, torch.device("cpu"))
        self.assertTrue(np.allclose(loaded.learned_thresholds().frame, 0.6))
        self.assertTrue(np.allclose(loaded.learned_thresholds().onset, model.learned_thresholds().onset))
        self.assertEqual(result["architecture"], "onsets-calibrated")
        self.assertEqual(torch.load(output, weights_only=True)["format_version"], 5)

    def test_recurrent_convolution_preserves_initial_predictions_and_learns_feedback(self):
        source = OnsetsPianoNet(hidden_size=16, gru_layers=1).eval()
        model = RecurrentPianoNet(hidden_size=16, gru_layers=1, conv_steps=3).eval()
        self.assertTrue(transfer_weights(model, {"model": source.state_dict()}))
        wave = torch.randn(1, 16000) * 0.1
        with torch.no_grad():
            original, recurrent = source(wave), model(wave)
        for head in original:
            self.assertTrue(torch.equal(original[head], recurrent[head]), head)
        block = RecurrentConvBlock(32, steps=3)
        features = torch.randn(1, 32, 16, 20, requires_grad=True)
        self.assertTrue(torch.equal(block(features), features))
        with torch.no_grad():
            block.scale.fill_(0.1)
        feedback_calls = []
        hook = block.feedback.register_forward_hook(lambda *args: feedback_calls.append(1))
        result = block(features)
        result.square().mean().backward()
        hook.remove()
        self.assertEqual(len(feedback_calls), 3)
        self.assertEqual(result.shape, features.shape)
        self.assertGreater(float(block.feedback.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(block.feedback.weight.grad).all())
        with self.assertRaises(ValueError):
            RecurrentPianoNet(conv_steps=1)
        options = Namespace(seed=42, epochs=1, positive_weight=None, architecture="onsets-recurrent",
                            conv_steps=0, device="cpu", hidden_size=16, gru_layers=1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            train(options)
        options.architecture, options.conv_steps = "onsets", 3
        with self.assertRaisesRegex(ValueError, "requires --architecture"):
            train(options)

    def test_recurrent_training_resume_and_frozen_thresholds(self):
        original = OnsetsPianoNet(hidden_size=16, gru_layers=1)
        source = self.root / "recognizer.pt"
        torch.save({"model": original.state_dict(), "model_config": original.config}, source)
        output = self.root / "recurrent.pt"
        options = Namespace(data=str(self.root), output=str(output), epochs=1,
                            batch_size=1, seconds=1.0, windows_per_file=2,
                            max_files=None, positive_weight=5.0, lr=1e-3,
                            seed=42, workers=0, device="cpu", resume=None,
                            reset_optimizer=False, architecture="onsets-recurrent", init_from=str(source),
                            freeze_thresholds=True, augment=False)
        train(options)
        latest = self.root / "recurrent.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["model_config"]["conv_steps"], 3)
        self.assertEqual(saved["model_config"]["architecture"], "onsets-recurrent")
        self.assertTrue(saved["training_config"]["freeze_thresholds"])
        self.assertTrue(torch.equal(saved["model"]["threshold_module.raw"], torch.zeros(3, 3)))
        self.assertTrue(any(float(saved["model"][f"recurrent_blocks.{stage}.scale"].abs()) > 0
                            for stage in ("stage2", "stage4")))
        options.resume, options.init_from, options.lr = str(latest), None, None
        options.freeze_thresholds = None
        train(options)
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["optimizer"]["param_groups"][1]["params"], [])
        self.assertTrue(torch.equal(saved["model"]["threshold_module.raw"], resumed["model"]["threshold_module.raw"]))
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu")
        self.assertEqual(result["architecture"], "onsets-recurrent")
        self.assertEqual(len(result["thresholds"]["frame"]), 88)

    def test_global_thresholds_learn_and_decode_all_pitches_with_the_same_cutoffs(self):
        module = GlobalThresholds()
        self.assertEqual(module.raw.shape, (3,))
        self.assertFalse(module.export().pitch_dependent)
        logits = torch.zeros(1, 30, 88, requires_grad=True)
        truth = torch.zeros_like(logits)
        truth[:, 5:10, (0, 48, 87)] = 1
        loss = module.loss({head: logits for head in ("frame", "onset", "offset")},
                           {head: truth for head in ("frame", "onset", "offset")})
        loss.backward()
        self.assertIsNone(logits.grad)
        self.assertTrue(torch.isfinite(module.raw.grad).all())
        self.assertTrue(torch.all(module.raw.grad != 0))
        module.initialize(Thresholds(frame=0.6, onset=0.7, offset=0.8))
        values = module.export()
        probabilities = {head: np.zeros((50, 88), np.float32) for head in ("frame", "onset", "offset")}
        for column in (0, 48, 87):
            probabilities["frame"][5:25, column] = 0.9
            probabilities["onset"][5, column] = 0.9
            probabilities["offset"][25, column] = 0.9
        notes = decode_outputs(probabilities, 1, values.frame,
                               onset_threshold=values.onset, offset_threshold=values.offset)["notes"]
        self.assertEqual([note["pitch"] for note in notes], [21, 69, 108])
        self.assertTrue(all((note["start"], note["end"]) == (0.1, 0.5) for note in notes))

    def test_v5_initialization_is_random_and_repeatable_with_an_explicit_seed(self):
        torch.manual_seed(123)
        first = GlobalRecurrentPianoNet(hidden_size=16, gru_layers=1)
        torch.manual_seed(123)
        repeat = GlobalRecurrentPianoNet(hidden_size=16, gru_layers=1)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, repeat.state_dict()[key]), key)
        torch.manual_seed(456)
        different = GlobalRecurrentPianoNet(hidden_size=16, gru_layers=1)
        self.assertFalse(torch.equal(first.onset_head.weight, different.onset_head.weight))
        self.assertFalse(torch.equal(first.threshold_module.raw, different.threshold_module.raw))
        self.assertTrue(all(float(block.scale.detach().abs()) > 0 for block in first.recurrent_blocks.values()))
        self.assertTrue(all(isinstance(value, float) for value in first.learned_thresholds().to_dict().values()))

    def test_v5_fresh_training_resume_scalar_thresholds_and_quiet_output(self):
        checkpoint = self.root / "v5.pt"
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=1,
                            batch_size=1, seconds=1.0, windows_per_file=1,
                            max_files=None, positive_weight=5.0, lr=1e-3,
                            seed=None, workers=0, device="cpu", resume=None,
                            reset_optimizer=False, architecture="onsets-recurrent-global", hidden_size=16, gru_layers=1,
                            augment=False)
        console = io.StringIO()
        with patch("piano_ml.training.secrets.randbits", return_value=123456) as choose_seed, redirect_stdout(console):
            train(options)
        choose_seed.assert_called_once_with(32)
        output = console.getvalue()
        self.assertIn("Fresh random initialization; seed=123456", output)
        self.assertIn("Learning 3 global thresholds", output)
        self.assertNotIn("Register thresholds:", output)
        self.assertNotIn('"bass"', output)
        self.assertNotIn('"low_note"', output)
        saved = torch.load(self.root / "v5.last.pt", weights_only=True)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["format_version"], 6)
        self.assertEqual(saved["model_config"]["architecture"], "onsets-recurrent-global")
        self.assertEqual(saved["model"]["threshold_module.raw"].shape, (3,))
        self.assertTrue(all(isinstance(value, float) for value in saved["thresholds"].values()))
        self.assertEqual(saved["training_config"]["seed"], 123456)
        self.assertEqual(saved["selection_metric"], "note_f_avg")
        self.assertEqual(saved["training_config"]["patience"], 0)
        self.assertIn("note_f_avg=", output)
        self.assertIn("stale_epochs=", output)
        self.assertIn("note_macro_f1=", output)
        self.assertIn("covered_pitches=", output)
        self.assertEqual(len(saved["validation"]["per_pitch_notes"]), 88)
        history = json.loads(checkpoint.with_suffix(".history.jsonl").read_text().strip())
        self.assertNotIn("register_thresholds", history)
        self.assertEqual(history["validation"]["per_pitch_notes"][0]["name"], "A0")
        self.assertEqual(history["validation"]["per_pitch_notes"][-1]["name"], "C8")
        # Older checkpoints have the same selection score but no pitch report.
        for key in ("note_macro_f1", "note_macro_pitch_count", "per_pitch_notes"):
            saved["validation"].pop(key, None)
        torch.save(saved, self.root / "v5.last.pt")
        options.resume = str(self.root / "v5.last.pt")
        options.lr = None
        with patch("piano_ml.training.secrets.randbits", side_effect=AssertionError("Resume must keep the seed")), redirect_stdout(io.StringIO()):
            train(options)
        resumed = torch.load(self.root / "v5.last.pt", weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["training_config"]["seed"], 123456)
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(len(resumed["validation"]["per_pitch_notes"]), 88)
        prediction = transcribe_audio(self.root / "test.wav", checkpoint, device="cpu")
        self.assertNotIn("register_thresholds", prediction)
        self.assertTrue(all(isinstance(value, float) for value in prediction["thresholds"].values()))
        calibrated = self.root / "v5-calibrated.pt"
        evaluation = Namespace(data=str(self.root), checkpoint=str(checkpoint), split="validation",
                               threshold=None, thresholds=None, frame_threshold=0.6, selection_metric="note_f1",
                               min_note_seconds=None, full_recordings=True, max_files=1,
                               calibrate_output=str(calibrated), output=str(self.root / "validation.json"), device="cpu")
        with redirect_stdout(io.StringIO()):
            report = evaluate(evaluation)
        self.assertEqual(report["note_macro_pitch_count"], 1)
        self.assertEqual(report["per_pitch_notes"][69 - 21]["reference_count"], 1)
        with (self.root / "validation.pitches.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 88)
        self.assertEqual(rows[69 - 21]["name"], "A4")
        self.assertEqual(rows[69 - 21]["reference_count"], "1")
        self.assertEqual(json.loads((self.root / "validation.json").read_text())["per_pitch_notes"], report["per_pitch_notes"])
        loaded = load_model(calibrated, torch.device("cpu"))
        self.assertAlmostEqual(loaded.learned_thresholds().frame, 0.6)
        self.assertEqual(torch.load(calibrated, weights_only=True)["format_version"], 6)

    def test_validation_download_expands_only_validation_and_respects_budget(self):
        csv_path = self.root / "maestro-v3.0.0.csv"
        rows = list(csv.DictReader(csv_path.open(newline="")))
        for index in range(3):
            rows.append({"split": "validation" if index < 2 else "train",
                         "midi_filename": f"extra-{index}.midi", "audio_filename": f"extra-{index}.wav",
                         "duration": "1.0"})
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        wav_bytes = (self.root / "validation.wav").read_bytes()
        archives = {}
        for kind, field in (("audio", "audio_filename"), ("midi", "midi_filename")):
            archive = self.root / f"{kind}.zip"
            with zipfile.ZipFile(archive, "w") as writer:
                for row in rows:
                    writer.writestr(row[field], wav_bytes if kind == "audio" else midi_bytes())
            archives[kind] = archive
        def remote_zip(url, **kwargs):
            return zipfile.ZipFile(archives["midi" if "-midi.zip" in url else "audio"])
        with patch.dict("sys.modules", {"remotezip": types.SimpleNamespace(RemoteZip=remote_zip)}):
            with self.assertRaisesRegex(ValueError, "within"):
                download_validation(self.root, total=3, gigabytes=0.000001)
            plan = download_validation(self.root, total=3, gigabytes=0.001, dry_run=True)
            self.assertEqual(plan["existing_recordings"], 1)
            self.assertEqual(plan["new_recordings"], 2)
            self.assertLess(plan["planned_additional_bytes"], plan["budget_bytes"])
            self.assertFalse((self.root / "extra-0.wav").exists())
            report = download_validation(self.root, total=3, gigabytes=0.001)
            self.assertEqual(report["available_validation_recordings"], 3)
            self.assertTrue(all(row["split"] == "validation" for row in report["records"]))
            self.assertFalse((self.root / "extra-2.wav").exists())
            for index in (0, 1):
                self.assertEqual((self.root / f"extra-{index}.wav").read_bytes(), wav_bytes)
            self.assertEqual(len(MaestroWindows(self.root, "validation").records), 3)

    def test_midi_velocity_pedal_and_multi_targets(self):
        events = bytes([0, 0x90, 69, 80, 0, 0xB0, 64, 127,
                        0x83, 0x60, 0x80, 69, 0, 0x83, 0x60, 0xB0, 64, 0,
                        0, 0xFF, 0x2F, 0])
        original = midi_bytes()
        (self.root / "train.midi").write_bytes(original[:18] + len(events).to_bytes(4, "big") + events)
        notes, pedals = read_performance(self.root / "train.midi")
        self.assertEqual(notes[0].velocity, 80)
        self.assertEqual((notes[0].end, pedals[0].end), (0.5, 1.0))
        _, targets = MaestroWindows(self.root, "train", seconds=1, windows_per_file=1,
                                   multi_target=True)[0]
        self.assertEqual(targets["onset"][0, 48], 1)
        self.assertAlmostEqual(float(targets["velocity"][0, 48]), 80 / 127)
        self.assertEqual(targets["offset"][25, 48], 1)
        self.assertEqual(targets["frame"][30, 48], 0)
        self.assertEqual(targets["pedal"][30, 0], 1)

    def test_note_metrics_require_unique_matches_and_offsets(self):
        reference = [{"pitch": 60, "start": 0.1, "end": 0.6},
                     {"pitch": 60, "start": 0.14, "end": 0.6}]
        prediction = [{"pitch": 60, "start": 0.12, "end": 0.6}]
        self.assertEqual(note_counts(reference, prediction), (1, 0, 1))
        prediction[0]["end"] = 0.9
        self.assertEqual(note_counts(reference, prediction), (0, 1, 2))
        self.assertEqual(note_counts(reference, prediction, offsets=False), (1, 0, 1))
        # An erroneous prediction reaching the clip edge still counts as an error.
        complete, guess = window_note_pairs(reference, [{"pitch": 60, "start": 0.2, "end": 1.0}], 1.0)
        self.assertEqual(note_counts(complete, guess), (0, 1, 2))

    def test_note_f0_to_f4_mean_and_empty_counts(self):
        scores = note_f_scores(2, 1, 3)
        expected = [2 / 3, 4 / 8, 10 / 23, 20 / 48, 34 / 83]
        for beta, value in enumerate(expected):
            self.assertAlmostEqual(scores[f"note_f{beta}"], value)
        self.assertAlmostEqual(scores["note_f_avg"], sum(expected) / 5)
        self.assertTrue(all(value == 1 for value in note_f_scores(3, 0, 0).values()))
        for counts in ((0, 0, 0), (0, 0, 5), (0, 3, 0)):
            self.assertTrue(all(value == 0 for value in note_f_scores(*counts).values()))

    def test_calibration_mean_can_select_a_different_model_than_note_f1(self):
        precise, more_recall = Thresholds(0.6), Thresholds(0.4)
        calibration = Calibration([precise, more_recall])
        reference = [{"pitch": pitch, "start": 0.1, "end": 0.4} for pitch in range(21, 31)]
        predictions = [reference[:5], reference[:8] + [
            {"pitch": pitch, "start": 0.1, "end": 0.4} for pitch in range(50, 58)]]
        probabilities = {head: np.zeros((50, 88), np.float32) for head in ("frame", "onset", "offset")}
        with patch("piano_ml.calibration.decode_outputs", side_effect=[{"notes": notes} for notes in predictions]):
            calibration.add(probabilities, np.zeros((50, 88), bool), reference, 1)
        f1 = calibration.results("note_f1")
        average = calibration.results()
        self.assertEqual(f1["thresholds"], precise.to_dict())
        self.assertEqual(average["thresholds"], more_recall.to_dict())
        self.assertLess(average["note_f1"], f1["note_f1"])
        self.assertGreater(average["note_f_avg"], f1["note_f_avg"])

    def test_cli_defaults_to_no_early_stopping(self):
        with patch("sys.argv", ["piano_ml", "train"]), patch("piano_ml.__main__.train") as run:
            cli_main()
        options = run.call_args.args[0]
        self.assertEqual(options.patience, 0)
        self.assertEqual(options.selection_metric, "auto")
        with patch("sys.argv", ["piano_ml", "train", "--patience", "20", "--reset-early-stopping"]), \
             patch("piano_ml.__main__.train") as run:
            cli_main()
        self.assertEqual(run.call_args.args[0].patience, 20)
        self.assertTrue(run.call_args.args[0].reset_early_stopping)

    def test_average_selection_preserves_best_and_completes_all_requested_epochs(self):
        checkpoint = self.root / "mean-selection.pt"
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=45,
                            batch_size=1, seconds=0.2, windows_per_file=1, max_files=None,
                            positive_weight=5.0, lr=1e-3, seed=42, workers=0, device="cpu",
                            resume=None, architecture="onsets", hidden_size=16, gru_layers=1,
                            augment=False, lr_patience=100)
        values = iter([0.6] + [0.5] * 19 + [0.65] + [0.64] * 24)
        def fake_score(*args, **kwargs):
            value = next(values)
            return {"loss": 0.1, "frame_bce": 0.1, "f1": 1 - value, "onset_f1": 1 - value,
                    "note_f1": 1 - value, "note_f_avg": value, "precision": 0.5,
                    "recall": 0.5, "threshold": 0.5, "threshold_curve": []}
        console = io.StringIO()
        with patch("piano_ml.training.score", side_effect=fake_score), redirect_stdout(console):
            train(options)
        best = torch.load(checkpoint, weights_only=True)
        latest = torch.load(self.root / "mean-selection.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 21)
        self.assertEqual(best["best_score"], 0.65)
        self.assertEqual(best["selection_metric"], "note_f_avg")
        self.assertEqual(latest["epoch"], 45)
        self.assertEqual(latest["stale_epochs"], 24)
        self.assertNotIn("Early stopping after", console.getvalue())
        history = [json.loads(line) for line in checkpoint.with_suffix(".history.jsonl").read_text().splitlines()]
        self.assertEqual(history[20]["stale_epochs"], 0)
        self.assertEqual(len(history), 45)
        self.assertEqual(history[-1]["patience"], 0)
        self.assertFalse(history[-1]["early_stopping"])

    def test_resume_old_f1_checkpoint_restarts_comparison_using_average(self):
        model = GlobalRecurrentPianoNet(hidden_size=16, gru_layers=1)
        checkpoint = self.root / "old-f1.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config,
                    "epoch": 7, "selection_metric": "note_f1", "best_score": 0.99,
                    "stale_epochs": 19}, checkpoint)
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=1,
                            batch_size=1, seconds=0.2, windows_per_file=1, max_files=None,
                            positive_weight=5.0, lr=1e-3, seed=42, workers=0, device="cpu",
                            resume=str(checkpoint), architecture="auto", augment=False)
        def fake_score(current, *args, **kwargs):
            thresholds = current.learned_thresholds()
            return {"loss": 0.1, "frame_bce": 0.1, "f1": 0.9, "onset_f1": 0.9, "note_f1": 0.9,
                    "note_f_avg": 0.6, "precision": 0.9, "recall": 0.9,
                    "threshold": thresholds.frame, "thresholds": thresholds.to_dict(), "threshold_curve": []}
        console = io.StringIO()
        with patch("piano_ml.training.score", side_effect=fake_score) as validation, redirect_stdout(console):
            train(options)
        self.assertEqual(validation.call_count, 2)  # Re-score the starting weights, then epoch 8.
        best = torch.load(checkpoint, weights_only=True)
        latest = torch.load(self.root / "old-f1.last.pt", weights_only=True)
        self.assertEqual(best["epoch"], 7)
        self.assertEqual(best["best_score"], 0.6)
        self.assertEqual(latest["stale_epochs"], 1)
        self.assertEqual(latest["selection_metric"], "note_f_avg")
        self.assertIn("resetting best-score comparison and non-improvement count", console.getvalue())

    def test_v2_transfer_training_resume_scheduler_and_calibration(self):
        legacy = self.root / "legacy.pt"
        torch.save({"model": PianoNet().state_dict()}, legacy)
        checkpoint = self.root / "v2.pt"
        options = Namespace(data=str(self.root), output=str(checkpoint), epochs=6,
                            batch_size=1, seconds=1.0, windows_per_file=1,
                            max_files=None, positive_weight=5.0, lr=1e-3,
                            seed=42, workers=0, device="cpu", resume=None,
                            reset_optimizer=False, architecture="onsets", init_from=str(legacy),
                            hidden_size=16, gru_layers=1, patience=2, lr_patience=0,
                            thresholds=[0.5], augment=False)
        def fake_score(value):
            return {"loss": 0.1, "frame_bce": 0.1, "f1": value, "onset_f1": value, "note_f1": value, "note_f_avg": value,
                    "precision": value, "recall": value, "threshold": 0.5, "threshold_curve": []}
        with patch("piano_ml.training.score", side_effect=[fake_score(0.5), fake_score(0.4)] + [fake_score(0.3)] * 4):
            train(options)
        latest = self.root / "v2.last.pt"
        saved = torch.load(latest, weights_only=True)
        self.assertEqual(saved["epoch"], 6)
        self.assertEqual(saved["stale_epochs"], 5)
        self.assertAlmostEqual(saved["optimizer"]["param_groups"][0]["lr"], 0.00003125)
        self.assertEqual(saved["selection_metric"], "note_f_avg")
        model = load_model(checkpoint, torch.device("cpu"))
        self.assertIsInstance(model, OnsetsPianoNet)
        output = model(torch.zeros(1, 16000))
        self.assertEqual(output["onset"].shape, (1, 51, 88))
        self.assertEqual(output["pedal"].shape, (1, 51, 1))
        options.resume, options.init_from, options.lr, options.epochs = str(latest), None, None, 1
        with patch("piano_ml.training.score", return_value=fake_score(0.6)):
            train(options)
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 7)
        self.assertAlmostEqual(resumed["optimizer"]["param_groups"][0]["lr"], 0.00003125)
        self.assertEqual(resumed["scheduler"]["last_epoch"], 7)
        calibrated = self.root / "calibrated.pt"
        evaluation = Namespace(data=str(self.root), checkpoint=str(checkpoint), split="validation",
                               thresholds=[0.35, 0.65], threshold=None, selection_metric="note_f1",
                               min_note_seconds=None, full_recordings=True, max_files=1,
                               calibrate_output=str(calibrated), output=None, device="cpu")
        result = evaluate(evaluation)
        self.assertEqual(len(result["threshold_curve"]), 2)
        self.assertEqual(load_model(calibrated, torch.device("cpu")).detection_threshold, result["threshold"])
        prediction = transcribe_audio(self.root / "test.wav", calibrated, device="cpu")
        self.assertEqual(prediction["architecture"], "onsets")
        self.assertIn("pedals", prediction)
        evaluation.split = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            evaluate(evaluation)

    def test_v2_context_keeps_note_crossing_chunk_boundary(self):
        audio = self.root / "long.wav"
        samples = np.zeros(128640, dtype="<i2")
        samples[63680], samples[67840] = 20000, -20000  # 3.98 s onset, 4.24 s offset
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(samples.tobytes())

        class FakeOnsets:
            architecture = "onsets"
            min_note_seconds = 0.04
            config = {"architecture": "onsets"}

            def __call__(self, wave):
                frames = wave.shape[1] // 320 + 1
                output = {key: torch.full((1, frames, 88), -20.0)
                          for key in ("frame", "onset", "offset", "velocity")}
                output["frame"][:, :, 39] = 4.0
                output["velocity"][:, :, 39] = 0
                output["pedal"] = torch.full((1, frames, 1), -20.0)
                for sample in torch.nonzero(wave[0]).flatten():
                    key = "onset" if wave[0, sample] > 0 else "offset"
                    output[key][0, sample // 320, 39] = 4.0
                return output

        for context in (None, 1, 2):
            for blend in ("crop", "weighted", "equal"):
                with self.subTest(context=context, blend=blend), patch("piano_ml.inference.load_model", return_value=FakeOnsets()):
                    result = transcribe_audio(audio, "fake.pt", device="cpu", context_seconds=context,
                                              overlap_blend=blend)
                    self.assertEqual(len(result["notes"]), 1)
                    self.assertEqual((result["notes"][0]["start"], result["notes"][0]["end"]), (3.98, 4.24))

    def test_result_playback_uses_pedal_and_velocity(self):
        result = {"duration": 1.2, "notes": [{"pitch": 69, "start": 0.1, "end": 0.4, "velocity": 100}],
                  "pedals": [{"start": 0.2, "end": 0.9}]}
        path = self.root / "pedal.wav"
        render_result_wav(result, path, engine="basic")
        with wave.open(str(path), "rb") as wav:
            loud = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        self.assertGreater(np.max(abs(loud[16000:17500])), 100)
        self.assertEqual(result["notes"][0]["end"], 0.4)
        result["notes"][0]["velocity"] = 25
        render_result_wav(result, path, engine="basic")
        with wave.open(str(path), "rb") as wav:
            quiet = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        self.assertLess(np.sqrt(np.mean(quiet ** 2)), np.sqrt(np.mean(loud ** 2)) * 0.6)

    def test_staff_positions_and_pagination(self):
        self.assertEqual(staff_pitch(60), ("treble", -1.0, False))  # middle C ledger line
        self.assertEqual(staff_pitch(64), ("treble", 0.0, False))   # treble bottom E4
        self.assertEqual(staff_pitch(77), ("treble", 4.0, False))   # treble top F5
        self.assertEqual(staff_pitch(43), ("bass", -10.0, False))   # bass bottom G2
        self.assertEqual(staff_pitch(57), ("bass", -6.0, False))    # bass top A3
        self.assertEqual(staff_pitch(66), ("treble", 0.5, True))
        self.assertEqual(note_value(0.5, 120), (1, 0, False))
        self.assertEqual(note_value(0.375, 120), (0.75, 1, True))
        result = {"duration": 9, "notes": [{"pitch": 60, "name": "C4", "start": 2,
                                            "end": 6, "confidence": 0.9}], "chords": []}
        figure = Figure(figsize=(10, 6))
        canvas = FigureCanvasAgg(figure)
        axis, page, pages, window = draw_staff(figure, result, bpm=120, page=1)
        canvas.draw()
        self.assertEqual((page, pages, window), (1, 3, (4.0, 8.0)))
        staff_lines = [line for line in axis.lines if np.allclose(line.get_xdata(), [-1.65, 8.15])]
        self.assertEqual(len(staff_lines), 10)

    def test_synthesized_pitch_timing_polyphony_and_cancellation(self):
        rate = 22050
        result = {"duration": 1.4, "notes": [{"pitch": 69, "start": 0.2, "end": 0.8}], "chords": []}
        output = self.root / "result.wav"
        render_result_wav(result, output, engine="basic")
        with wave.open(str(output), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, rate))
            self.assertEqual(wav.getnframes(), round((1.4 + RELEASE_SECONDS) * rate))
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        self.assertEqual(np.max(np.abs(audio[:int(0.19 * rate)])), 0)
        self.assertEqual(np.max(np.abs(audio[int(0.99 * rate):])), 0)
        segment = audio[int(0.25 * rate):int(0.65 * rate)]
        spectrum = abs(np.fft.rfft(segment * np.hanning(len(segment))))
        frequencies = np.fft.rfftfreq(len(segment), 1 / rate)
        self.assertAlmostEqual(frequencies[np.argmax(spectrum)], 440, delta=3)
        result["notes"].append({"pitch": 72, "start": 0.2, "end": 0.8})
        render_result_wav(result, output, engine="basic")
        with wave.open(str(output), "rb") as wav:
            chord = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        segment = chord[int(0.25 * rate):int(0.65 * rate)].astype(np.float32)
        spectrum = abs(np.fft.rfft(segment * np.hanning(len(segment))))
        for frequency in (440, 523.25):
            index = np.argmin(abs(frequencies - frequency))
            self.assertGreater(spectrum[index], np.max(spectrum) * 0.3)
        self.assertLess(np.max(abs(chord.astype(np.int32))), 32767)
        before = output.read_bytes()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            render_result_wav(result, output, cancel=cancel, engine="basic")
        self.assertEqual(output.read_bytes(), before)
        self.assertFalse(output.with_name("result.wav.partial").exists())


if __name__ == "__main__":
    unittest.main()
