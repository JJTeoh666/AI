import io
import unittest

import torch
from torch.nn import functional as F

from piano_ml.model import (DEFAULT_ARCHITECTURE, V11_ARCHITECTURE,
                            STFTFANSustainPianoNet, STFTCQTPianoNet,
                            ThreeBranchPianoNet, FourierRecurrentPianoNet,
                            build_model, checkpoint_format_version,
                            default_selection_metric, model_config)


class V11ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_budget_order_four_second_alignment_and_all_gradients(self):
        model = STFTFANSustainPianoNet()
        self.assertEqual(model.architecture, V11_ARCHITECTURE)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10256830)
        self.assertEqual(model.temporal.num_layers, 4)
        self.assertTrue(model.temporal.bidirectional)
        self.assertEqual(model.raw_encoder.window_size, 8192)
        self.assertEqual(model.threshold_module.betas, (1, 2, 3, 4, 5))
        self.assertEqual(default_selection_metric(V11_ARCHITECTURE), "avgf_loss")
        self.assertEqual(checkpoint_format_version(V11_ARCHITECTURE), 13)
        calls, inputs = [], {}
        hooks = [getattr(model, key).register_forward_pre_hook(
            lambda module, args, name=key: (calls.append(name), inputs.update({name: args[0].shape})) and None)
            for key in ("short_encoder", "raw_encoder", "temporal", "residual", "pedal_temporal")]
        outputs = model(torch.randn(1, 64000) * .1)
        for hook in hooks:
            hook.remove()
        self.assertEqual(inputs["short_encoder"], (1, 1, 1025, 201))
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

    def test_long_fan_context_hann_amplitude_and_short_odd_alignment(self):
        model = STFTFANSustainPianoNet(feature_width=16, hidden_size=8,
                                       gru_layers=1, pedal_hidden_size=4).eval()
        wave = torch.zeros(1, 16000)
        wave[0, 8000] = .25
        frames = model.raw_encoder.raw_frames(wave)
        self.assertEqual(frames.shape, (1, 51, 8192))
        self.assertAlmostEqual(float(frames[0, 25, 4096]), .25)
        torch.testing.assert_close(model.raw_encoder.raw_frames(wave * 3), frames * 3)
        self.assertGreater(int((frames.abs().sum(-1) > 0).sum()), 20)
        # Both representations are centered at sample 25 * 320 = 8000.
        self.assertEqual(int(model.short_spectrum(wave)[0, 56].argmax()), 25)
        self.assertEqual(int(frames.abs().amax(-1)[0].argmax()), 25)
        for samples in (1, 319, 320, 321, 16001):
            with torch.no_grad():
                outputs = model(torch.randn(1, samples) * .1)
            for head, value in outputs.items():
                self.assertEqual(value.shape,
                                 (1, samples // 320 + 1, 1 if head.startswith("pedal") else 88))
                self.assertTrue(torch.isfinite(value).all())

    def test_configuration_checkpoint_roundtrip_and_old_parameter_budgets(self):
        config = dict(architecture=V11_ARCHITECTURE, feature_width=16, hidden_size=8,
                      gru_layers=2, pedal_hidden_size=4, fan_window=4096, short_fft=1024)
        model = build_model(config).eval()
        self.assertEqual(model.note_duration, "sustain_extended")
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
        for model_class, parameter_count in ((STFTCQTPianoNet, 3054988),
                                              (ThreeBranchPianoNet, 10168060),
                                              (FourierRecurrentPianoNet, 10012924)):
            legacy = model_class()
            self.assertEqual(sum(p.numel() for p in legacy.parameters()), parameter_count)

    def test_invalid_dimensions_and_duration_semantics_rejected(self):
        for options in ({"feature_width": 17}, {"hidden_size": 10}, {"gru_layers": 0},
                        {"fan_window": 3000}, {"fan_window": 1024}, {"short_fft": 3000},
                        {"pedal_hidden_size": 0}, {"residual_blocks": 0},
                        {"note_duration": "key_release"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                STFTFANSustainPianoNet(**options)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_default_cuda_forward_backward_optimizer(self):
        model = STFTFANSustainPianoNet().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        outputs = model(torch.randn(1, 64000, device="cuda") * .1)
        targets = {head: torch.zeros_like(value) for head, value in outputs.items()}
        for head in targets:
            targets[head][:, 30:70] = .7 if head == "velocity" else 1
        loss = sum(F.binary_cross_entropy_with_logits(outputs[head], target)
                   for head, target in targets.items())
        (loss + model.threshold_module.loss(outputs, targets)).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        original = model.raw_encoder.fan.periodic.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(original, model.raw_encoder.fan.periodic.weight))


if __name__ == "__main__":
    unittest.main()
