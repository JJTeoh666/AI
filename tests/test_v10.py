import io
import json
import math
import os
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
from piano_ml.cqt import ConstantQ, HarmonicStack, HARMONICS, cqt_basis
from piano_ml.evaluation import evaluate
from piano_ml.inference import load_model, predict_probabilities, transcribe_audio
from piano_ml.model import (V10_ARCHITECTURE, HarmonicCQTPianoNet,
                            ThreeBranchPianoNet, FourierRecurrentPianoNet,
                            build_model, default_selection_metric)
from piano_ml.training import train, training_loss
from tests import test_v8 as v8, test_v9 as v9


class V10FrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cqt = ConstantQ()

    def test_fft_projection_matches_direct_variable_length_wavelets(self):
        generator = torch.Generator().manual_seed(123)
        audio = torch.randn(2, 16001, generator=generator) * 0.2
        for index, group in enumerate(self.cqt.groups):
            frequencies = self.cqt.frequencies[index * 36:(index + 1) * 36].tolist()
            basis = cqt_basis(frequencies, group.n_fft)
            windows = torch.nn.functional.pad(audio, (group.n_fft // 2, group.n_fft // 2))
            windows = windows.unfold(-1, group.n_fft, 320)
            direct = torch.einsum("btn,pqn->btpq", windows, basis)
            expected = torch.linalg.vector_norm(direct, dim=-1).transpose(1, 2)
            actual = group(audio)
            with self.subTest(octave=index):
                # Sparse Fourier truncation is checked against the full filters.
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-4)

    def test_tones_amplitude_silence_and_shared_frame_centers(self):
        time = torch.arange(32000) / 16000
        for midi in (21, 48, 69, 96, 108):
            frequency = 440 * 2 ** ((midi - 69) / 12)
            tone = (0.2 * torch.sin(2 * torch.pi * frequency * time))[None]
            result = self.cqt(tone)
            center = result[0, :, 50]
            self.assertEqual(int(center.argmax()), (midi - 21) * 3)
            self.assertAlmostEqual(float(center.max()), 0.1, delta=0.003)
            torch.testing.assert_close(self.cqt(tone * 3), result * 3, atol=1e-6, rtol=1e-4)
        for length in (1, 319, 320, 321, 16001, 64000):
            result = self.cqt(torch.zeros(1, length))
            self.assertEqual(result.shape, (1, 295, length // 320 + 1))
            self.assertEqual(float(result.abs().sum()), 0)
        impulse = torch.zeros(1, 32000)
        impulse[0, 16000] = 1
        # All nine FFT groups center the impulse at the same 20 ms position.
        for group in self.cqt.groups:
            self.assertEqual(int(group(impulse)[0, 0].argmax()), 50)

    def test_harmonic_channels_gather_correct_bins_and_mask_missing_energy(self):
        stack = HarmonicStack(295)
        data = torch.arange(295 * 4, dtype=torch.float32).reshape(1, 295, 4)
        result = stack(data)
        self.assertEqual(result.shape, (1, 8, 264, 4))
        for channel, harmonic in enumerate(HARMONICS):
            shift = round(36 * math.log2(harmonic))
            for candidate in (0, 36, 150, 261, 263):
                source = candidate + shift
                expected = data[0, source] if 0 <= source < 295 else torch.zeros(4)
                torch.testing.assert_close(result[0, channel, candidate], expected)
        self.assertEqual(sum(p.numel() for p in self.cqt.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in stack.parameters()), 0)


class V10ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_budget_alignment_and_all_parameters_receive_finite_gradients(self):
        model = HarmonicCQTPianoNet()
        count = lambda part: sum(p.numel() for p in part.parameters() if p.requires_grad)
        self.assertEqual(default_selection_metric(V10_ARCHITECTURE), "avgf_loss")
        self.assertEqual(count(model), 3035788)
        for name, expected in (("cqt_encoder", 407112), ("long_encoder", 386904),
                               ("fusion", 80600), ("residual", 1445400), ("temporal", 549888)):
            self.assertEqual(count(getattr(model, name)), expected)
        branches = {}
        handles = [getattr(model, name).register_forward_hook(
            lambda module, inputs, output, key=name: branches.update({key: output.shape}))
            for name in ("cqt_encoder", "long_encoder")]
        outputs = model(torch.randn(1, 64000) * 0.1)
        for handle in handles:
            handle.remove()
        self.assertEqual(set(branches.values()), {torch.Size((1, 201, 200))})
        targets = {head: torch.zeros_like(output) for head, output in outputs.items()}
        for head, output in outputs.items():
            self.assertEqual(output.shape, (1, 201, 1 if head == "pedal" else 88))
        for pitch, first, last in ((3, 10, 25), (40, 70, 120), (75, 150, 180)):
            targets["frame"][:, first:last, pitch] = 1
            targets["onset"][:, first:first + 2, pitch] = 1
            targets["offset"][:, last:last + 2, pitch] = 1
            targets["velocity"][:, first:first + 2, pitch] = 0.7
        targets["pedal"][:, 60:130] = 1
        loss = training_loss(outputs, targets, offset_loss_weight=1, release_loss_weight=0.25)
        (loss + model.threshold_module.loss(outputs, targets)).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(model.threshold_module.betas, (0, 1, 2, 3))
        self.assertFalse(model.learned_thresholds().pitch_dependent)

    def test_short_and_odd_audio_silence_backward_and_configuration_guards(self):
        model = HarmonicCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        for samples in (1, 319, 320, 321, 16001):
            with torch.no_grad():
                outputs = model(torch.randn(1, samples) * 0.1)
            for head, output in outputs.items():
                self.assertEqual(output.shape, (1, samples // 320 + 1, 1 if head == "pedal" else 88))
                self.assertTrue(torch.isfinite(output).all())
        outputs = model(torch.zeros(1, 16000))
        targets = {head: torch.zeros_like(output) for head, output in outputs.items()}
        (training_loss(outputs, targets) + model.threshold_module.loss(outputs, targets)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        # Top CQT bins survive pooling: ceil(264 / 16) = 17 positions.
        self.assertEqual(model.cqt_encoder.projection[0].in_features, 96 * 17)
        for width in (0, 8, 17, 31, 16.0):
            with self.assertRaisesRegex(ValueError, "divisible by eight"):
                HarmonicCQTPianoNet(feature_width=width)
        for option in ({"hidden_size": 0}, {"gru_layers": 0}, {"long_fft": 3000},
                       {"release_frames": 1}, {"dropout": 1}, {"rnn_backend": "bad"}):
            with self.assertRaises(ValueError):
                HarmonicCQTPianoNet(**option)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_default_cuda_batch_four_optimizer_step_and_float32_frontend(self):
        model = HarmonicCQTPianoNet().cuda()
        audio = torch.randn(4, 64000, device="cuda") * 0.1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            spectrum = model.cqt(audio)
        self.assertEqual(spectrum.dtype, torch.float32)
        expected = ConstantQ()(audio[:1].cpu())
        torch.testing.assert_close(spectrum[:1].cpu(), expected, atol=2e-6, rtol=1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        outputs = model(audio)
        targets = {head: torch.zeros_like(output) for head, output in outputs.items()}
        targets["frame"][:, 20:80, 40] = 1
        targets["onset"][:, 20:22, 40] = 1
        targets["offset"][:, 80:82, 40] = 1
        targets["velocity"][:, 20:22, 40] = 0.7
        targets["pedal"][:, 30:70] = 1
        loss = training_loss(outputs, targets) + model.threshold_module.loss(outputs, targets)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        before = model.frame_head.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(model.frame_head.weight, before))


class V10PipelineTest(unittest.TestCase):
    setUp = v8.V8PipelineTest.setUp
    tearDown = v8.V8PipelineTest.tearDown
    test_minimum_avgf_selection_scheduler_and_all_requested_epochs = v9.V9PipelineTest.test_avgf_selection_scheduler_and_no_early_stop
    test_resume_restores_random_and_optimizer_states_exactly = v9.V9PipelineTest.test_resume_restores_random_and_optimizer_states_exactly

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, output, **changes):
        return v8.V8PipelineTest.options(self, output, **{
            "architecture": V10_ARCHITECTURE, "fourier_modes": None, "fourier_layers": None, **changes})

    def test_explicit_v10_save_load_resume_initialization_and_loss_reporting(self):
        previous_directory = Path.cwd()
        console = io.StringIO()
        try:
            os.chdir(self.root)
            args = self.options(None, feature_width=None, hidden_size=None, gru_layers=None)
            args.output = None
            with redirect_stdout(console):
                train(args)
        finally:
            os.chdir(previous_directory)
        best = self.root / "checkpoints/piano-v10.pt"
        latest = self.root / "checkpoints/piano-v10.last.pt"
        history = self.root / "checkpoints/piano-v10.history.jsonl"
        saved = torch.load(latest, weights_only=True)
        self.assertTrue(best.exists())
        self.assertEqual(saved["format_version"], 11)
        self.assertEqual(saved["model_version"], "10")
        self.assertEqual(saved["model_config"]["architecture"], V10_ARCHITECTURE)
        self.assertEqual(saved["model_config"]["feature_width"], 200)
        self.assertEqual(saved["model_config"]["hidden_size"], 128)
        self.assertEqual(saved["model_config"]["gru_layers"], 2)
        self.assertEqual(saved["selection_metric"], "avgf_loss")
        self.assertEqual(saved["selection_mode"], "min")
        config = saved["training_config"]
        self.assertEqual(config["model_version"], "10")
        self.assertEqual(config["threshold_objective"], "pitch_f0123")
        self.assertFalse(config["early_stopping"])
        for key, value in (("bass_sampling", 0), ("middle_sampling", 0), ("treble_sampling", 0),
                           ("edge_loss_weight", 1), ("middle_loss_weight", 1),
                           ("offset_loss_weight", 1), ("release_loss_weight", 0.25)):
            self.assertEqual(config[key], value)
        record = json.loads(history.read_text())
        self.assertAlmostEqual(sum(record["train_loss_components"].values()), record["train_loss"], places=6)
        self.assertEqual(saved["training_metrics"]["frame_bce"], record["train_frame_bce"])
        self.assertIn("frame_bce", saved["validation"])
        self.assertIn("avgf_loss", saved["validation"])
        self.assertIn("Training V10", console.getvalue())
        loaded = load_model(latest, torch.device("cpu"))
        self.assertEqual(sum(p.numel() for p in loaded.parameters()), 3035788)
        recreated = build_model(saved["model_config"]).eval()
        recreated.load_state_dict(saved["model"])
        with torch.no_grad():
            audio = torch.randn(1, 16001) * 0.1
            before, after = loaded(audio), recreated(audio)
        for head in before:
            torch.testing.assert_close(before[head], after[head], atol=0, rtol=0)
        with redirect_stdout(console):
            train(self.options(best, resume=str(latest), lr=None, hidden_size=None,
                               gru_layers=None, feature_width=None))
        resumed = torch.load(latest, weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["model_config"], saved["model_config"])
        self.assertEqual(resumed["validation_signature"], saved["validation_signature"])
        self.assertEqual(resumed["scheduler"]["last_epoch"], saved["scheduler"]["last_epoch"] + 1)
        self.assertEqual(len(history.read_text().splitlines()), 2)
        with redirect_stdout(console):
            train(self.options(self.root / "initialized.pt", init_from=str(latest),
                               feature_width=None, hidden_size=None, gru_layers=None))
        self.assertIn("Transferred complete recognizer", console.getvalue())
        self.assertEqual(torch.load(self.root / "initialized.last.pt", weights_only=True)["epoch"], 1)
        result = transcribe_audio(self.root / "test.wav", latest, device="cpu", overlap_blend="equal")
        self.assertEqual(result["model_version"], "10")
        self.assertEqual(result["inference_config"]["overlap_blend"], "equal")

    def test_explicit_cli_and_reject_cross_architecture_initialization(self):
        with patch("sys.argv", ["piano_ml", "train", "--architecture", V10_ARCHITECTURE]), \
                patch("piano_ml.__main__.train") as run:
            main()
        self.assertEqual(run.call_args.args[0].architecture, V10_ARCHITECTURE)
        for model in (ThreeBranchPianoNet(feature_width=16, hidden_size=8, gru_layers=1),
                      FourierRecurrentPianoNet(feature_width=32, hidden_size=16, gru_layers=1)):
            source = self.root / "older.pt"
            torch.save({"model": model.state_dict(), "model_config": model.config}, source)
            with self.assertRaisesRegex(ValueError, "older model weights cannot transfer"):
                train(self.options(self.root / "bad.pt", init_from=str(source)))
        for option in ("fourier_modes", "fourier_layers"):
            with self.assertRaisesRegex(ValueError, "specific to V8.1"):
                train(self.options(self.root / "bad.pt", **{option: 2}))

    def test_evaluation_metadata_calibrated_copy_and_full_recordings(self):
        model = HarmonicCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.root / "source.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        self.assertEqual(load_model(source, torch.device("cpu")).model_version, "10")
        args = Namespace(data=str(self.root), checkpoint=str(source), split="validation",
                         threshold=None, thresholds=None, frame_threshold=0.55, selection_metric="auto",
                         full_recordings=False, min_note_seconds=None, seconds=1, max_files=None,
                         windows_per_file=1, batch_size=1, workers=0, device="cpu",
                         calibrate_output=str(self.root / "calibrated.pt"), output=str(self.root / "report.json"))
        with redirect_stdout(io.StringIO()):
            report = evaluate(args)
            args.full_recordings = True
            full = evaluate(args)
        self.assertEqual(report["model_version"], "10")
        self.assertEqual(report["selection_mode"], "min")
        self.assertEqual(full["selection_metric"], "avgf_loss")
        self.assertTrue(math.isfinite(full["avgf_loss"]))
        self.assertEqual(torch.load(args.calibrate_output, weights_only=True)["format_version"], 11)
        self.assertEqual(load_model(args.calibrate_output, torch.device("cpu")).model_version, "10")

    def test_all_overlap_methods_partial_recordings_and_cancellation(self):
        path = self.root / "boundary.wav"
        duration = 8.13
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(np.zeros(round(duration * 16000), dtype="<i2").tobytes())
        model = HarmonicCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1).eval()
        zero_overlap = []
        for context in (0, 1):
            for mode in ("crop", "weighted", "equal"):
                measured, probabilities = predict_probabilities(path, model, torch.device("cpu"),
                                                                context_seconds=context, overlap_blend=mode)
                self.assertAlmostEqual(measured, duration)
                for head, values in probabilities.items():
                    self.assertEqual(values.shape, (407, 1 if head == "pedal" else 88))
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(((values >= 0) & (values <= 1)).all())
                if context == 0:
                    zero_overlap.append(probabilities)
        for other in zero_overlap[1:]:
            for head in zero_overlap[0]:
                np.testing.assert_array_equal(other[head], zero_overlap[0][head])
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            predict_probabilities(path, model, torch.device("cpu"), cancel=cancel, overlap_blend="equal")


if __name__ == "__main__":
    unittest.main()
