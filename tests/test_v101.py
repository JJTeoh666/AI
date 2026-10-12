import argparse
import io
import json
import os
import re
import shlex
import threading
import unittest
import wave
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.__main__ import main
from piano_ml.inference import load_model, predict_probabilities, transcribe_audio
from piano_ml.evaluation import evaluate
from piano_ml.model import (DEFAULT_ARCHITECTURE, V101_ARCHITECTURE, V10_ARCHITECTURE,
                            STFTCQTPianoNet, HarmonicCQTPianoNet, build_model)
from piano_ml.training import train, training_loss
from tests import test_v8 as v8, test_v9 as v9


class V101ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_budget_branch_inputs_four_second_alignment_and_all_gradients(self):
        model = STFTCQTPianoNet()
        self.assertEqual(DEFAULT_ARCHITECTURE, V101_ARCHITECTURE)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3054988)
        self.assertEqual(sum(p.numel() for p in model.short_encoder.parameters()), 406104)
        self.assertEqual(sum(p.numel() for p in model.cqt_encoder.parameters()), 407112)
        inputs = {}
        hooks = [getattr(model, key).register_forward_pre_hook(
            lambda module, args, name=key: inputs.update({name: args[0].shape}))
            for key in ("short_encoder", "harmonic_stack", "cqt_encoder")]
        outputs = model(torch.randn(1, 64000) * .1)
        for hook in hooks:
            hook.remove()
        self.assertEqual(inputs["short_encoder"], (1, 1, 1025, 201))
        self.assertEqual(inputs["harmonic_stack"], (1, 295, 201))
        self.assertEqual(inputs["cqt_encoder"], (1, 8, 264, 201))
        truth = {head: torch.zeros_like(value) for head, value in outputs.items()}
        for head, value in outputs.items():
            self.assertEqual(value.shape, (1, 201, 1 if head == "pedal" else 88))
        for pitch, first, last in ((3, 10, 25), (40, 70, 120), (75, 150, 180)):
            truth["frame"][:, first:last, pitch] = 1
            truth["onset"][:, first:first + 2, pitch] = 1
            truth["offset"][:, last:last + 2, pitch] = 1
            truth["velocity"][:, first:first + 2, pitch] = .7
        truth["pedal"][:, 60:130] = 1
        loss = training_loss(outputs, truth, offset_loss_weight=1, release_loss_weight=.25)
        (loss + model.threshold_module.loss(outputs, truth)).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(model.threshold_module.betas, (0, 1, 2, 3))

    def test_stft_bins_hann_normalization_and_shared_frame_centers(self):
        model = STFTCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        time = torch.arange(32000) / 16000
        audio = (.2 * torch.sin(2 * torch.pi * 437.5 * time))[None]
        spectrum = model.short_spectrum(audio)
        self.assertEqual(int(spectrum[0, :, 50].argmax()), 56)
        self.assertAlmostEqual(float(spectrum[0, 56, 50]), .1, places=5)
        torch.testing.assert_close(model.short_spectrum(audio * 3), spectrum * 3, atol=1e-6, rtol=1e-3)
        impulse = torch.zeros_like(audio)
        impulse[0, 16000] = 1
        self.assertEqual(int(model.short_spectrum(impulse)[0, 56].argmax()), 50)
        self.assertEqual(int(model.cqt(impulse)[0, 56].argmax()), 50)
        for samples in (1, 319, 320, 321, 16001):
            with torch.no_grad():
                outputs = model(torch.randn(1, samples) * .1)
            for head, value in outputs.items():
                self.assertEqual(value.shape, (1, samples // 320 + 1, 1 if head == "pedal" else 88))
                self.assertTrue(torch.isfinite(value).all())
        outputs = model(torch.zeros(1, 16000))
        truth = {head: torch.zeros_like(value) for head, value in outputs.items()}
        (training_loss(outputs, truth) + model.threshold_module.loss(outputs, truth)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for option in ({"feature_width": 17}, {"short_fft": 3000}, {"short_fft": 256},
                       {"hidden_size": 0}, {"gru_layers": 0}, {"release_frames": 1}):
            with self.assertRaises(ValueError):
                STFTCQTPianoNet(**option)
        configured = STFTCQTPianoNet(short_fft=4096, feature_width=16, hidden_size=8, gru_layers=1)
        self.assertEqual(configured.short_spectrum(audio).shape, (1, 2049, 101))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_default_cuda_batch_four_optimizer_step(self):
        model = STFTCQTPianoNet().cuda()
        audio = torch.randn(4, 64000, device="cuda") * .1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            self.assertEqual(model.short_spectrum(audio).dtype, torch.float32)
            self.assertEqual(model.cqt(audio).dtype, torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        outputs = model(audio)
        truth = {head: torch.zeros_like(value) for head, value in outputs.items()}
        truth["frame"][:, 20:80, 40] = 1
        truth["onset"][:, 20:22, 40] = 1
        truth["offset"][:, 80:82, 40] = 1
        truth["velocity"][:, 20:22, 40] = .7
        truth["pedal"][:, 30:70] = 1
        (training_loss(outputs, truth) + model.threshold_module.loss(outputs, truth)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        before = model.frame_head.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.frame_head.weight))


class V101PipelineTest(unittest.TestCase):
    setUp = v8.V8PipelineTest.setUp
    tearDown = v8.V8PipelineTest.tearDown
    test_selection_scheduling_and_requested_epochs = v9.V9PipelineTest.test_avgf_selection_scheduler_and_no_early_stop
    test_resume_restores_optimizer_scheduler_and_random_states = v9.V9PipelineTest.test_resume_restores_random_and_optimizer_states_exactly

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, output, **changes):
        return v8.V8PipelineTest.options(self, output, **{
            "architecture": "auto", "fourier_modes": None, "fourier_layers": None, **changes})

    def test_fresh_auto_default_save_load_resume_and_compatible_initialization(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            args = self.options(None, feature_width=None, hidden_size=None, gru_layers=None)
            args.output = None
            with redirect_stdout(io.StringIO()):
                train(args)
        finally:
            os.chdir(previous)
        best = self.root / "checkpoints/piano-v10.1.pt"
        latest = self.root / "checkpoints/piano-v10.1.last.pt"
        history = self.root / "checkpoints/piano-v10.1.history.jsonl"
        saved = torch.load(latest, weights_only=True)
        self.assertTrue(best.exists())
        self.assertEqual(saved["format_version"], 12)
        self.assertEqual(saved["model_version"], "10.1")
        self.assertEqual(saved["model_config"]["architecture"], V101_ARCHITECTURE)
        self.assertEqual(saved["model_config"]["short_fft"], 2048)
        self.assertNotIn("long_fft", saved["model_config"])
        self.assertEqual(saved["model_config"]["feature_width"], 200)
        self.assertEqual(saved["selection_metric"], "avgf_loss")
        self.assertEqual(saved["selection_mode"], "min")
        self.assertFalse(saved["training_config"]["early_stopping"])
        self.assertEqual(saved["training_config"]["threshold_objective"], "pitch_f0123")
        record = json.loads(history.read_text())
        self.assertAlmostEqual(sum(record["train_loss_components"].values()), record["train_loss"], places=6)
        self.assertEqual(record["train_frame_bce"], saved["training_metrics"]["frame_bce"])
        loaded = load_model(latest, torch.device("cpu"))
        self.assertEqual(sum(p.numel() for p in loaded.parameters()), 3054988)
        recreated = build_model(saved["model_config"]).eval()
        recreated.load_state_dict(saved["model"])
        with torch.no_grad():
            audio = torch.randn(1, 16001) * .1
            first, second = loaded(audio), recreated(audio)
        for head in first:
            torch.testing.assert_close(first[head], second[head], atol=0, rtol=0)
        with redirect_stdout(io.StringIO()):
            train(self.options(best, resume=str(latest), lr=None, feature_width=None,
                               hidden_size=None, gru_layers=None))
            train(self.options(self.root / "initialized.pt", init_from=str(latest),
                               feature_width=None, hidden_size=None, gru_layers=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["model_config"], saved["model_config"])
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["scheduler"]["last_epoch"], saved["scheduler"]["last_epoch"] + 1)
        self.assertEqual(torch.load(self.root / "initialized.last.pt", weights_only=True)["epoch"], 1)
        self.assertEqual(len(history.read_text().splitlines()), 2)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu", overlap_blend="equal")
        self.assertEqual(result["model_version"], "10.1")

    def test_cli_options_and_rejected_v10_initialization(self):
        with patch("sys.argv", ["piano_ml", "train", "--architecture", V101_ARCHITECTURE,
                                "--short-fft", "4096"]), patch("piano_ml.__main__.train") as run:
            main()
        self.assertEqual(run.call_args.args[0].short_fft, 4096)
        source = self.root / "v10.pt"
        model = HarmonicCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        with self.assertRaisesRegex(ValueError, "V10.1 uses short STFT"):
            train(self.options(self.root / "bad.pt", init_from=str(source)))
        with self.assertRaisesRegex(ValueError, "replaces the long FFT"):
            train(self.options(self.root / "bad.pt", long_fft=8192))
        with self.assertRaisesRegex(ValueError, "specific to V10.1"):
            train(self.options(self.root / "bad.pt", architecture=V10_ARCHITECTURE, short_fft=2048))

    def test_all_overlap_methods_and_cancellation(self):
        path = self.root / "boundary.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(bytes(round(8.13 * 16000) * 2))
        model = STFTCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1).eval()
        for mode in ("crop", "weighted", "equal"):
            duration, outputs = predict_probabilities(path, model, torch.device("cpu"),
                                                      context_seconds=1, overlap_blend=mode)
            self.assertAlmostEqual(duration, 8.13)
            for head, values in outputs.items():
                self.assertEqual(values.shape, (407, 1 if head == "pedal" else 88))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(((values >= 0) & (values <= 1)).all())
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            predict_probabilities(path, model, torch.device("cpu"), cancel=cancel)

    def test_evaluation_and_calibrated_checkpoint_preserve_version_and_format(self):
        model = STFTCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        self.assertEqual(load_model(source, torch.device("cpu")).model_version, "10.1")
        args = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                         threshold=None, thresholds=None, frame_threshold=.55, selection_metric="auto",
                         full_recordings=False, min_note_seconds=None, seconds=1, max_files=None,
                         windows_per_file=1, batch_size=1, workers=0, device="cpu",
                         calibrate_output=str(self.root / "calibrated.pt"), output=str(self.root / "report.json"))
        with redirect_stdout(io.StringIO()):
            report = evaluate(args)
            args.full_recordings = True
            full = evaluate(args)
        self.assertEqual(report["model_version"], "10.1")
        self.assertEqual(report["selection_mode"], "min")
        self.assertTrue(np.isfinite(full["avgf_loss"]))
        self.assertEqual(torch.load(args.calibrate_output, weights_only=True)["format_version"], 12)
        self.assertEqual(load_model(args.calibrate_output, torch.device("cpu")).model_version, "10.1")


class ReadmeCommandsTest(unittest.TestCase):
    def test_commands_are_tables_and_cli_examples_parse_without_executing(self):
        readme = Path(__file__).resolve().parents[1] / "README.md"
        text = readme.read_text(encoding="utf-8")
        self.assertNotIn("```powershell", text)
        commands = re.findall(r"`(python -m piano_ml [^`\n]+)`", text)
        self.assertGreaterEqual(len(commands), 30)
        original = argparse.ArgumentParser.parse_args
        def parse_only(parser, *args, **kwargs):
            parsed = original(parser, *args, **kwargs)
            parsed.func = lambda _args: None
            return parsed
        for command in commands:
            with self.subTest(command=command), patch("sys.argv", shlex.split(command, posix=False)[2:]), \
                    patch.object(argparse.ArgumentParser, "parse_args", parse_only):
                main()
        for version in ("10.1", "10", "9", "8.1"):
            self.assertIn(f"--output checkpoints/piano-v{version}.pt", text)
            self.assertIn(f"--resume checkpoints/piano-v{version}.last.pt", text)
        inventory = json.loads((readme.parent / "diagnostics/v101-readme-commands.json").read_text(encoding="utf-8"))
        for command in inventory["previous_commands"]:
            self.assertIn(f"`{command}`", text)


if __name__ == "__main__":
    unittest.main()
