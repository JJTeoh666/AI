"""One-time, self-contained ONNX export for server-free browser inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import FOURIER_ARCHITECTURE, V9_ARCHITECTURE, build_model
from .thresholds import saved_thresholds


class RealFourierMixing(nn.Module):
    """Equivalent low-mode DFT written with ONNX-supported real operators."""

    def __init__(self, original):
        super().__init__()
        self.padding, self.modes = original.padding, original.modes
        self.dc_weight, self.band_weight = original.dc_weight, original.band_weight

    def forward(self, sequence):
        padded = nn.functional.pad(sequence.transpose(1, 2), (self.padding, self.padding)).transpose(1, 2)
        length = padded.shape[1]
        times = torch.arange(length, device=sequence.device, dtype=sequence.dtype)
        modes = torch.arange(self.modes, device=sequence.device, dtype=sequence.dtype)
        angle = times[:, None] * modes[None, :] * (2 * torch.pi / length)
        cosine, sine = angle.cos(), angle.sin()
        norm = torch.sqrt(torch.ones((), device=sequence.device) * length)
        real = torch.einsum("tm,btd->bmd", cosine, padded) / norm
        imag = -torch.einsum("tm,btd->bmd", sine, padded) / norm
        dc = real[:, :1] @ self.dc_weight
        wr, wi = self.band_weight[..., 0], self.band_weight[..., 1]
        mixed_real = torch.cat((dc, torch.einsum("bmi,mio->bmo", real[:, 1:], wr)
                               - torch.einsum("bmi,mio->bmo", imag[:, 1:], wi)), dim=1)
        mixed_imag = torch.cat((torch.zeros_like(dc), torch.einsum("bmi,mio->bmo", real[:, 1:], wi)
                               + torch.einsum("bmi,mio->bmo", imag[:, 1:], wr)), dim=1)
        factor = torch.where((modes == 0) | (2 * modes == length), 1., 2.) * (2 * modes <= length)
        result = (torch.einsum("tm,bmd->btd", cosine, mixed_real * factor[None, :, None])
                  - torch.einsum("tm,bmd->btd", sine, mixed_imag * factor[None, :, None])) / norm
        return result[:, self.padding:-self.padding] if self.padding else result


class BrowserPianoNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if model.architecture == FOURIER_ARCHITECTURE:
            for block in model.fourier:
                block.spectral = RealFourierMixing(block.spectral)

    def forward(self, short_mel, long_mel, raw_frames=None):
        model = self.model
        short = model.short_encoder(short_mel)
        long = model.long_encoder(long_mel)
        if model.architecture == V9_ARCHITECTURE:
            raw = model.raw_encoder.norm(model.raw_encoder.fan(raw_frames))
            raw = model.raw_encoder.convolution(raw.transpose(1, 2)).transpose(1, 2)
            features = model.residual(model.fusion(torch.cat((short, long, raw), dim=-1)))
        else:
            features = model.fourier(model.fusion(torch.cat((short, long), dim=-1)))
        sequence, _ = model.temporal(features)
        sequence = model.sequence_norm(sequence)
        onset = model.onset_head(sequence)
        frame = model.frame_head(torch.cat((sequence, onset.sigmoid()), dim=-1))
        pedal = model.pedal_head(sequence)
        context = torch.cat((sequence, frame.sigmoid(), onset.sigmoid(), pedal.sigmoid()), dim=-1)
        offset = model.offset_head(sequence) + model.offset_refinement(context)
        return tuple(value.sigmoid() for value in (frame, onset, offset, model.velocity_head(sequence), pedal))


def export_browser_model(checkpoint, output):
    import onnx
    import onnxruntime as ort

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved.get("model_config"))
    model.load_state_dict(saved["model"])
    model.model_version = str(saved.get("model_version", "8" if model.architecture == FOURIER_ARCHITECTURE else "9"))
    model.decoding_thresholds = model.learned_thresholds() if hasattr(model, "learned_thresholds") else saved_thresholds(saved)
    model.min_note_seconds = float(saved.get("min_note_seconds", .04))
    model.eval()
    if model.architecture not in (FOURIER_ARCHITECTURE, V9_ARCHITECTURE):
        raise ValueError("Browser export currently supports V8/V8.1 and V9 checkpoints.")
    output = Path(output)
    if output.suffix.lower() != ".onnx":
        raise ValueError("Choose an .onnx output filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "piano-browser-v1", "architecture": model.architecture,
                "version": model.model_version, "parameters": sum(p.numel() for p in model.parameters()),
                "sample_rate": 16000, "hop_length": 320, "long_fft": model.config["long_fft"],
                "release_frames": model.config.get("release_frames", 3),
                "min_note_seconds": model.min_note_seconds,
                "thresholds": model.decoding_thresholds.to_dict()}
    # Sparse exact filter-bank coefficients avoid rounding differences from regenerating HTK banks.
    for name in ("short", "long"):
        bank = getattr(model, name + "_mel").mel_scale.fb.numpy()
        metadata[name + "_window"] = getattr(model, name + "_mel").spectrogram.window.tolist()
        metadata[name + "_bank"] = [[[int(i), float(bank[i, m])] for i in np.flatnonzero(bank[:, m])]
                                     for m in range(bank.shape[1])]
    generator = torch.Generator().manual_seed(923)
    verification_waves = [torch.randn(1, count, generator=generator) * .1 for count in (1600, 64000, 96000)]
    with torch.inference_mode():
        original_outputs = [[value.sigmoid().numpy() for value in model(wave).values()] for wave in verification_waves]
    wrapper = BrowserPianoNet(model).eval()
    samples = torch.randn(1, 64000) * 0.1
    def features(wave):
        values = tuple(torch.log1p(getattr(model, name + "_mel")(wave).clamp_min(0)) for name in ("short", "long"))
        return values + (model.raw_encoder.raw_frames(wave),) if model.architecture == V9_ARCHITECTURE else values
    inputs = features(samples)
    names = ["short_mel", "long_mel"] + (["raw_frames"] if len(inputs) == 3 else [])
    heads = ["frame", "onset", "offset", "velocity", "pedal"]
    dynamic = {name: {2: "frames"} for name in names[:2]}
    if len(inputs) == 3:
        dynamic["raw_frames"] = {1: "frames"}
    dynamic.update({name: {1: "frames"} for name in heads})
    temporary = output.with_name(output.name + ".partial")
    try:
        with torch.inference_mode():
            torch.onnx.export(wrapper, inputs, str(temporary), input_names=names, output_names=heads,
                              dynamic_axes=dynamic, opset_version=17, dynamo=False, external_data=False)
        graph = onnx.load(str(temporary))
        onnx.helper.set_model_props(graph, {"piano_metadata": json.dumps(metadata, separators=(",", ":"))})
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
        errors = []
        # Different lengths exercise dynamic GRU, DFT and the extra centered FFT endpoint.
        for wave, original in zip(verification_waves, original_outputs):
            count = wave.shape[1]
            args = features(wave)
            with torch.inference_mode():
                expected = [value.numpy() for value in wrapper(*args)]
            actual = session.run(heads, {name: value.numpy() for name, value in zip(names, args)})
            for head, wanted, got, native in zip(heads, expected, actual, original):
                if wanted.shape != got.shape or not np.allclose(wanted, got, rtol=2e-4, atol=3e-5):
                    raise ValueError(f"ONNX verification failed for {head} at {count} samples.")
                if not np.allclose(native, got, rtol=3e-4, atol=4e-5):
                    raise ValueError(f"Browser probability differs from the original PyTorch {head}.")
                errors.append(float(np.max(np.abs(wanted - got))))
        temporary.replace(output)
        return {"output": str(output.resolve()), "version": metadata["version"], "bytes": output.stat().st_size,
                "maximum_probability_error": max(errors)}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Export a V8.1 or V9 checkpoint for the server-free Piano Notes Site.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        torch.set_num_threads(2)
        print(json.dumps(export_browser_model(args.checkpoint, args.output), indent=2))
    except ImportError:
        parser.exit(1, "Install browser export dependencies first: python -m pip install -r requirements-browser.txt\n")


if __name__ == "__main__":
    main()
