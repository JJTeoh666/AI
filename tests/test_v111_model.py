import io
import unittest

import torch
from torch.nn import functional as F

from piano_ml.model import (V111_ARCHITECTURE,
                            DualSTFTSustainPianoNet, STFTFANSustainPianoNet,
                            STFTCQTPianoNet, ThreeBranchPianoNet,
                            FourierRecurrentPianoNet, FourierAnalysisLayer,
                            build_model, checkpoint_format_version,
                            default_selection_metric, model_config)


class V111ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_budget_branch_inputs_order_alignment_and_all_gradients(self):
        model = DualSTFTSustainPianoNet()
        self.assertEqual(model.architecture, V111_ARCHITECTURE)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9929470)
        self.assertEqual(sum(p.numel() for p in model.short_encoder.parameters()), 706944)
        self.assertEqual(sum(p.numel() for p in model.long_encoder.parameters()), 2476416)
        self.assertFalse(any(isinstance(module, FourierAnalysisLayer) for module in model.modules()))
        self.assertFalse(hasattr(model, "raw_encoder"))
        self.assertEqual(model.temporal.num_layers, 4)
        self.assertTrue(model.temporal.bidirectional)
        self.assertEqual(model.threshold_module.betas, (1, 2, 3, 4, 5))
        self.assertEqual(default_selection_metric(V111_ARCHITECTURE), "avgf_loss")
        self.assertEqual(checkpoint_format_version(V111_ARCHITECTURE), 14)
        calls, inputs = [], {}

        def remember(name):
            def hook(module, args):
                calls.append(name)
                inputs[name] = args[0].shape
            return hook

        hooks = [getattr(model, key).register_forward_pre_hook(remember(key))
                 for key in ("short_encoder", "long_encoder", "temporal", "residual", "pedal_temporal")]
        outputs = model(torch.randn(1, 64000) * .1)
        for hook in hooks:
            hook.remove()
        self.assertEqual(inputs["short_encoder"], (1, 1, 1025, 201))
        self.assertEqual(inputs["long_encoder"], (1, 1, 4097, 201))
        self.assertEqual(inputs["temporal"], (1, 201, 384))
        self.assertEqual(inputs["residual"], (1, 201, 448))
        self.assertLess(calls.index("temporal"), calls.index("residual"))
        self.assertEqual(set(outputs), {"frame", "onset", "offset", "velocity",
                                        "pedal", "pedal_onset", "pedal_offset"})
        targets = {head: torch.zeros_like(value) for head, value in outputs.items()}
        for head, value in outputs.items():
            self.assertEqual(value.shape, (1, 201, 1 if head.startswith("pedal") else 88))
            targets[head][:, 30:70] = .7 if head == "velocity" else 1
        loss = sum(F.binary_cross_entropy_with_logits(outputs[head], target)
                   for head, target in targets.items())
        (loss + model.threshold_module.loss(outputs, targets)).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_spectrum_frequency_amplitude_context_and_short_odd_alignment(self):
        model = DualSTFTSustainPianoNet(feature_width=16, hidden_size=8,
                                       gru_layers=1, pedal_hidden_size=4).eval()
        time = torch.arange(32000) / 16000
        wave = (.2 * torch.sin(2 * torch.pi * 437.5 * time))[None]
        short, long = model.short_spectrum(wave), model.long_spectrum(wave)
        self.assertEqual(short.shape, (1, 1025, 101))
        self.assertEqual(long.shape, (1, 4097, 101))
        self.assertEqual(int(short[0, :, 50].argmax()), 56)
        self.assertEqual(int(long[0, :, 50].argmax()), 224)
        self.assertAlmostEqual(float(short[0, 56, 50]), .1, places=5)
        self.assertAlmostEqual(float(long[0, 224, 50]), .1, places=5)
        torch.testing.assert_close(model.short_spectrum(wave * 3), short * 3, atol=1e-6, rtol=1e-3)
        torch.testing.assert_close(model.long_spectrum(wave * 3), long * 3, atol=1e-6, rtol=1e-3)
        impulse = torch.zeros_like(wave)
        impulse[0, 16000] = 1
        short_impulse, long_impulse = model.short_spectrum(impulse), model.long_spectrum(impulse)
        self.assertEqual(int(short_impulse[0, 56].argmax()), 50)
        self.assertEqual(int(long_impulse[0, 224].argmax()), 50)
        self.assertGreater(int((long_impulse[0, 224] > 0).sum()),
                           int((short_impulse[0, 56] > 0).sum()))
        for samples in (1, 319, 320, 321, 16001):
            with torch.no_grad():
                outputs = model(torch.randn(1, samples) * .1)
            for head, value in outputs.items():
                self.assertEqual(value.shape,
                                 (1, samples // 320 + 1, 1 if head.startswith("pedal") else 88))
                self.assertTrue(torch.isfinite(value).all())

    def test_configuration_checkpoint_roundtrip_and_old_parameter_budgets(self):
        config = dict(architecture=V111_ARCHITECTURE, feature_width=16, hidden_size=8,
                      gru_layers=2, pedal_hidden_size=4, long_fft=4096, short_fft=1024)
        model = build_model(config).eval()
        self.assertEqual(model.note_duration, "sustain_extended")
        self.assertNotIn("fan_window", model_config(model))
        buffer = io.BytesIO()
        torch.save({"model_config": model_config(model), "model": model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model(checkpoint["model_config"]).eval()
        restored.load_state_dict(checkpoint["model"])
        wave = torch.randn(1, 16001) * .1
        with torch.no_grad():
            expected, actual = model(wave), restored(wave)
        for head in expected:
            torch.testing.assert_close(expected[head], actual[head], atol=0, rtol=0)
        for model_class, parameter_count in ((STFTFANSustainPianoNet, 10256830),
                                              (STFTCQTPianoNet, 3054988),
                                              (ThreeBranchPianoNet, 10168060),
                                              (FourierRecurrentPianoNet, 10012924)):
            legacy = model_class()
            self.assertEqual(sum(p.numel() for p in legacy.parameters()), parameter_count)
        old = STFTFANSustainPianoNet(feature_width=16, hidden_size=8, gru_layers=1,
                                    pedal_hidden_size=4)
        old_restored = build_model(model_config(old))
        old_restored.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(old.state_dict()), set(old_restored.state_dict()))

    def test_invalid_dimensions_and_fan_option_rejected(self):
        for options in ({"feature_width": 17}, {"hidden_size": 10}, {"gru_layers": 0},
                        {"long_fft": 3000}, {"long_fft": 1024}, {"short_fft": 3000},
                        {"pedal_hidden_size": 0}, {"residual_blocks": 0},
                        {"note_duration": "key_release"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DualSTFTSustainPianoNet(**options)
        with self.assertRaises(TypeError):
            DualSTFTSustainPianoNet(fan_window=8192)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_default_cuda_forward_backward_optimizer_and_float32_fft(self):
        model = DualSTFTSustainPianoNet().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        audio = torch.randn(1, 64000, device="cuda") * .1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            self.assertEqual(model.short_spectrum(audio).dtype, torch.float32)
            self.assertEqual(model.long_spectrum(audio).dtype, torch.float32)
        outputs = model(audio)
        targets = {head: torch.zeros_like(value) for head, value in outputs.items()}
        for head in targets:
            targets[head][:, 30:70] = .7 if head == "velocity" else 1
        loss = sum(F.binary_cross_entropy_with_logits(outputs[head], target)
                   for head, target in targets.items())
        (loss + model.threshold_module.loss(outputs, targets)).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        original = model.long_encoder.projection[0].weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(original, model.long_encoder.projection[0].weight))


if __name__ == "__main__":
    unittest.main()
