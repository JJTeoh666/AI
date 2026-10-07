"""Shared inference for the command line and desktop viewer."""

import math
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .data import read_wav_window
from .decode import decode, decode_outputs
from .model import HOP_LENGTH, SAMPLE_RATE, FOURIER_ARCHITECTURE, build_model, model_config
from .thresholds import saved_thresholds, Thresholds


CHUNK_SECONDS = 4.0
MAX_CONTEXT_SECONDS = 2.0


def inference_context(model, context_seconds: float | None = None) -> float:
    """Resolve context on each side, aligned to the prediction's 20 ms grid."""
    if context_seconds is None:
        context_seconds = 0.5 if getattr(model, "architecture", "frame") != "frame" else 0.0
    if not math.isfinite(context_seconds) or not 0 <= context_seconds <= MAX_CONTEXT_SECONDS:
        raise ValueError("Analysis context must be between 0 and 2 seconds on each side.")
    return round(round(context_seconds * SAMPLE_RATE / HOP_LENGTH) * HOP_LENGTH / SAMPLE_RATE, 6)


def get_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Choose Auto or CPU.")
    return torch.device(choice)


def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved.get("model_config"))
    model.load_state_dict(saved["model"])
    if model.architecture == FOURIER_ARCHITECTURE:
        # Legacy V8 files have identical tensors but no minor-version metadata.
        model.model_version = str(saved.get("model_version", "8"))
    model.decoding_thresholds = model.learned_thresholds() if hasattr(model, "learned_thresholds") else saved_thresholds(saved)
    model.detection_threshold = model.decoding_thresholds.summary()["frame"]
    model.min_note_seconds = float(saved.get("min_note_seconds", 0.04 if model.architecture != "frame" else 0.08))
    return model.to(device).eval()


def transcribe_audio(audio: str | Path, checkpoint: str | Path, threshold: float | None = None,
                     device: str = "auto", progress: Callable[[int, int], None] | None = None,
                     cancel: threading.Event | None = None,
                     min_note_seconds: float | None = None,
                     frame_threshold: float | None = None,
                     onset_threshold: float | None = None,
                     offset_threshold: float | None = None,
                     context_seconds: float | None = None,
                     overlap_blend: str = "crop") -> dict:
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if overlap_blend not in ("crop", "weighted", "equal"):
        raise ValueError("Overlap blend must be 'crop', 'weighted', or 'equal'.")
    selected_device = get_device(device)
    model = load_model(checkpoint, selected_device)
    context = inference_context(model, context_seconds)
    base = getattr(model, "decoding_thresholds", Thresholds(
        frame=getattr(model, "detection_threshold", 0.5), onset=getattr(model, "detection_threshold", 0.5)))
    values = base.override(shared=threshold, frame=frame_threshold, onset=onset_threshold, offset=offset_threshold)
    threshold = values.summary()["frame"]
    minimum = getattr(model, "min_note_seconds", 0.08) if min_note_seconds is None else min_note_seconds
    duration, probabilities = predict_probabilities(audio, model, selected_device, progress, cancel,
                                                   context_seconds=context, overlap_blend=overlap_blend)
    # Preserve the legacy decoding path for existing models.
    events = (decode(probabilities["frame"], duration, threshold=threshold, min_note_seconds=minimum)
              if len(probabilities) == 1 else decode_outputs(probabilities, duration, threshold, minimum,
                  frame_threshold=values.frame, onset_threshold=values.onset, offset_threshold=values.offset,
                  release_frames=model_config(model).get("release_frames", 2)))
    return {"audio": str(Path(audio).resolve()), "model": str(Path(checkpoint).resolve()),
            "architecture": model_config(model)["architecture"],
            **({"model_version": model.model_version} if hasattr(model, "model_version") else {}),
            "duration": round(duration, 6), "threshold": threshold,
            "thresholds": values.to_dict(),
            "inference_config": {"chunk_seconds": CHUNK_SECONDS, "context_seconds": context,
                                 "overlap_seconds": round(2 * context, 6), "overlap_blend": overlap_blend},
            **({"register_thresholds": values.registers()} if values.pitch_dependent else {}),
            "min_note_seconds": minimum, **events}


def predict_probabilities(audio: str | Path, model: torch.nn.Module, device: torch.device,
                          progress=None, cancel=None,
                          context_seconds: float | None = None,
                          overlap_blend: str = "crop") -> tuple[float, dict[str, np.ndarray]]:
    if overlap_blend not in ("crop", "weighted", "equal"):
        raise ValueError("Overlap blend must be 'crop', 'weighted', or 'equal'.")
    context = inference_context(model, context_seconds)
    with wave.open(str(audio), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Choose an uncompressed 16-bit PCM WAV recording.")
        duration = wav.getnframes() / wav.getframerate()
    if duration <= 0:
        raise ValueError("Audio file is empty.")
    chunk_seconds = CHUNK_SECONDS
    chunk_frames = round(chunk_seconds * SAMPLE_RATE / HOP_LENGTH)
    total = math.ceil(duration / chunk_seconds)
    all_probabilities = {}
    context_frames = round(context * SAMPLE_RATE / HOP_LENGTH)
    blending = overlap_blend != "crop" and context_frames > 0
    frame_count = math.ceil(duration * SAMPLE_RATE / HOP_LENGTH - 1e-9)
    weight_sum = np.zeros(frame_count, dtype=np.float32) if blending else None
    with torch.inference_mode():
        for chunk in range(total):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Analysis cancelled.")
            start = chunk * chunk_seconds
            input_start = max(0, start - context)
            samples = read_wav_window(audio, input_start, chunk_seconds + start - input_start + context)
            wave_tensor = torch.from_numpy(samples).unsqueeze(0).to(device)
            output = model(wave_tensor)
            output = output if isinstance(output, dict) else {"frame": output}
            if blending:
                start_frame = chunk * chunk_frames
                first = max(0, start_frame - context_frames)
                # Exclude the input's extra STFT endpoint and all silence beyond the WAV.
                last = min(frame_count, start_frame + chunk_frames + context_frames)
                frames = np.arange(first, last)
                # Unit weights make 'equal' the mean of the available section probabilities.
                weights = np.ones(last - first, dtype=np.float32)
                if overlap_blend == "weighted" and chunk > 0:
                    weights *= np.clip((frames - (start_frame - context_frames)) / (2 * context_frames), 0, 1)
                if overlap_blend == "weighted" and chunk < total - 1:
                    weights *= np.clip((start_frame + chunk_frames + context_frames - frames) / (2 * context_frames), 0, 1)
                if chunk and output.keys() != all_probabilities.keys():
                    raise ValueError("Model output heads changed between analysis sections.")
                for key, logits in output.items():
                    probabilities = logits.sigmoid()[0].cpu().numpy()
                    if probabilities.ndim != 2 or len(probabilities) < len(weights):
                        raise ValueError(f"Model output '{key}' does not cover the analysis section.")
                    if key not in all_probabilities:
                        all_probabilities[key] = np.zeros((frame_count, probabilities.shape[1]), dtype=np.float32)
                    all_probabilities[key][first:last] += probabilities[:len(weights)] * weights[:, None]
                weight_sum[first:last] += weights
            else:
                # STFT includes the endpoint at t=4 s. The next chunk owns that
                # frame; keeping it twice would make note timestamps drift.
                remaining_frames = math.ceil((duration - start) * SAMPLE_RATE / HOP_LENGTH - 1e-9)
                valid_frames = min(chunk_frames, remaining_frames)
                first = round((start - input_start) * SAMPLE_RATE / HOP_LENGTH)
                for key, logits in output.items():
                    probabilities = logits.sigmoid()[0].cpu().numpy()
                    all_probabilities.setdefault(key, []).append(probabilities[first:first + valid_frames])
            if progress is not None:
                progress(chunk + 1, total)
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Analysis cancelled.")
    if blending:
        if np.any(weight_sum <= 0):
            raise RuntimeError("Overlapping analysis left uncovered prediction frames.")
        for probabilities in all_probabilities.values():
            probabilities /= weight_sum[:, None]
        return duration, all_probabilities
    return duration, {key: np.concatenate(chunks) for key, chunks in all_probabilities.items()}
