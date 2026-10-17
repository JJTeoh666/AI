import io
import os
import unittest

import torch
from torch.nn import functional as F

from piano_ml.model import (FIVE_SCORE_ARCHITECTURES,
                            V12_ARCHITECTURE, SplitFANSustainPianoNet,
                            DualSTFTSustainPianoNet, STFTFANSustainPianoNet,
                            STFTCQTPianoNet, ThreeBranchPianoNet,
                            FourierRecurrentPianoNet, FourierAnalysisLayer,
                            build_model, checkpoint_format_version,
                            default_selection_metric, model_config)


class V12ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def small(**options):
        config = dict(feature_width=16, hidden_size=8, gru_layers=1,
                      event_width=16, event_fan_layers=2, dropout=0)
        config.update(options)
        return SplitFANSustainPianoNet(**config)

    @staticmethod
    def objective(model, outputs):
        targets = {head: torch.zeros_like(value) for head, value in outputs.items()}
        for head in targets:
            targets[head][:, 30:70] = .7 if head == "velocity" else 1
        return sum(F.binary_cross_entropy_with_logits(outputs[head], target)
                   for head, target in targets.items()) + model.threshold_module.loss(outputs, targets)

    def test_default_budget_routing_alignment_and_all_gradients(self):
        model = SplitFANSustainPianoNet()
        self.assertEqual(model.architecture, V12_ARCHITECTURE)
        self.assertIn(V12_ARCHITECTURE, FIVE_SCORE_ARCHITECTURES)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9991654)
        expected = {"short_encoder": 706944, "long_encoder": 2476416,
                    "fusion": 296064, "temporal": 2168832,
                    "sequence_norm": 1024, "residual": 3150336,
                    "frame_head": 45144, "velocity_head": 45144,
                    "event_fan": 964096, "onset_head": 67672,
                    "offset_head": 67672, "pedal_head": 769,
                    "pedal_onset_head": 769, "pedal_offset_head": 769,
                    "threshold_module": 3}
        for key, count in expected.items():
            self.assertEqual(sum(p.numel() for p in getattr(model, key).parameters()), count, key)
        self.assertEqual(model.temporal.num_layers, 2)
        self.assertEqual(model.temporal.hidden_size, 256)
        self.assertTrue(model.temporal.bidirectional)
        self.assertFalse(hasattr(model, "pedal_temporal"))
        self.assertFalse(hasattr(model, "offset_refinement"))
        self.assertEqual(sum(isinstance(layer, FourierAnalysisLayer)
                             for layer in model.event_fan), 2)
        self.assertEqual(model.threshold_module.betas, (1, 2, 3, 4, 5))
        self.assertEqual(default_selection_metric(V12_ARCHITECTURE), "avgf_loss")
        self.assertEqual(checkpoint_format_version(V12_ARCHITECTURE), 15)
        observed = {}

        def remember(name):
            def hook(module, args):
                observed[name] = tuple(args[0].shape)
            return hook

        hooks = [getattr(model, key).register_forward_pre_hook(remember(key))
                 for key in ("short_encoder", "long_encoder", "temporal", "residual", "event_fan")]
        outputs = model(torch.randn(1, 64000) * .1)
        for hook in hooks:
            hook.remove()
        self.assertEqual(observed["short_encoder"], (1, 1, 1025, 201))
        self.assertEqual(observed["long_encoder"], (1, 1, 4097, 201))
        self.assertEqual(observed["temporal"], (1, 201, 384))
        self.assertEqual(observed["residual"], (1, 201, 512))
        self.assertEqual(observed["event_fan"], (1, 201, 896))
        self.assertEqual(set(outputs), {"frame", "onset", "offset", "velocity",
                                        "pedal", "pedal_onset", "pedal_offset"})
        for head, value in outputs.items():
            self.assertEqual(value.shape, (1, 201, 1 if head.startswith("pedal") else 88))
        self.objective(model, outputs).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        original = model.event_fan[1].periodic.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(original, model.event_fan[1].periodic.weight))

    def test_direct_short_connection_and_separate_head_paths(self):
        model = self.small().eval()
        captured = {}

        def keep_short(module, args, value):
            captured["short"] = value

        def keep_events(module, args):
            captured["events"] = args[0]

        hooks = [model.short_encoder.register_forward_hook(keep_short),
                 model.event_fan.register_forward_pre_hook(keep_events),
                 model.temporal.register_forward_hook(
                     lambda module, args, value: (torch.zeros_like(value[0]), value[1]))]
        wave = torch.randn(1, 3200) * .1
        first = model(wave)
        torch.testing.assert_close(captured["events"][..., 16:], captured["short"], atol=0, rtol=0)
        self.assertEqual(float(captured["events"][..., :16].detach().abs().sum()), 0)
        second = model(wave * 2)
        torch.testing.assert_close(first["frame"], second["frame"], atol=0, rtol=0)
        torch.testing.assert_close(first["velocity"], second["velocity"], atol=0, rtol=0)
        self.assertFalse(torch.equal(first["onset"], second["onset"]))
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        result = model(wave)
        (result["onset"].square().mean() + result["pedal"].square().mean()).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.residual.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.frame_head.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.velocity_head.parameters()))
        self.assertGreater(float(model.short_encoder.projection[0].weight.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        model(wave)["frame"].square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.event_fan.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.onset_head.parameters()))
        self.assertGreater(float(model.residual[0].convolution[0].weight.grad.abs().sum()), 0)

    def test_spectrum_amplitude_context_alignment_and_mixed_precision(self):
        model = self.small().eval()
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
        self.assertGreater(int((long_impulse[0, 224] > 0).sum()),
                           int((short_impulse[0, 56] > 0).sum()))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(model.short_spectrum(wave).dtype, torch.float32)
            self.assertEqual(model.long_spectrum(wave).dtype, torch.float32)
        for samples in (1, 319, 320, 321, 16001):
            with torch.no_grad():
                outputs = model(torch.randn(1, samples) * .1)
            for head, value in outputs.items():
                self.assertEqual(value.shape,
                                 (1, samples // 320 + 1, 1 if head.startswith("pedal") else 88))
                self.assertTrue(torch.isfinite(value).all())

    def test_checkpoint_roundtrip_and_legacy_models(self):
        model = self.small(gru_layers=2, event_width=20, event_fan_layers=3,
                           long_fft=4096, short_fft=1024).eval()
        self.assertEqual(model.note_duration, "sustain_extended")
        config = model_config(model)
        self.assertEqual(config["event_width"], 20)
        self.assertEqual(config["event_fan_layers"], 3)
        self.assertNotIn("fan_window", config)
        self.assertNotIn("pedal_hidden_size", config)
        buffer = io.BytesIO()
        torch.save({"model_config": config, "model": model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model(checkpoint["model_config"]).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        wave = torch.randn(1, 16001) * .1
        with torch.no_grad():
            expected, actual = model(wave), restored(wave)
        for head in expected:
            torch.testing.assert_close(expected[head], actual[head], atol=0, rtol=0)
        for model_class, count in ((DualSTFTSustainPianoNet, 9929470),
                                  (STFTFANSustainPianoNet, 10256830),
                                  (STFTCQTPianoNet, 3054988),
                                  (ThreeBranchPianoNet, 10168060),
                                  (FourierRecurrentPianoNet, 10012924)):
            legacy = model_class()
            self.assertEqual(sum(p.numel() for p in legacy.parameters()), count)
            old = build_model(model_config(legacy))
            old.load_state_dict(legacy.state_dict(), strict=True)

    def test_invalid_options(self):
        for options in ({"feature_width": 17}, {"hidden_size": 10}, {"gru_layers": 0},
                        {"long_fft": 3000}, {"long_fft": 1024}, {"short_fft": 3000},
                        {"residual_blocks": 0}, {"event_width": 0}, {"event_width": 3},
                        {"event_fan_layers": 0}, {"note_duration": "key_release"},
                        {"rnn_backend": "other"}, {"release_frames": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.small(**options)
        with self.assertRaises(TypeError):
            self.small(fan_window=8192)

    @unittest.skipUnless(os.environ.get("PIANO_V12_GPU_TEST") == "1" and torch.cuda.is_available(),
                         "Explicit idle-GPU test not requested")
    def test_default_cuda_forward_backward(self):
        model = SplitFANSustainPianoNet().cuda()
        audio = torch.randn(1, 64000, device="cuda") * .1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            self.assertEqual(model.short_spectrum(audio).dtype, torch.float32)
            self.assertEqual(model.long_spectrum(audio).dtype, torch.float32)
        self.objective(model, model(audio)).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
