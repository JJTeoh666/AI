"""Offline ByteDance note-and-pedal comparison using the official checkpoint.

Architecture/decoder provenance and licenses: piano_ml/_bytedance/NOTICE.
The app adapter uses upstream 10-second windows and 50% center stitching.
"""

import hashlib
import math
import threading
import time
import wave
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .decode import chord_name, pitch_name

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets/bytedance/note-pedal.pth"
MODEL_BYTES = 171966578
MODEL_MD5 = "22b961b77c1878239fec963362097045"
MODEL_SHA256 = "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141"
MODEL_URL = "https://zenodo.org/api/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth/content"
SOURCE_COMMIT = "1ade7dcd4348add669a67c6e6282456c8c6633bd"
SAMPLE_RATE, FPS, SEGMENT_SAMPLES, STRIDE_SAMPLES = 16000, 100, 160000, 80000
THRESHOLDS = {"onset": .3, "offset": .3, "frame": .1, "pedal_offset": .2, "pedal_frame": .5}
OUTPUT_WIDTHS = {"reg_onset_output": 88, "reg_offset_output": 88, "frame_output": 88,
                 "velocity_output": 88, "reg_pedal_onset_output": 1,
                 "reg_pedal_offset_output": 1, "pedal_frame_output": 1}


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("ByteDance analysis cancelled.")


def verify_checkpoint(path=MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("ByteDance model is missing. Run: python diagnostics/download_bytedance.py")
    if path.stat().st_size != MODEL_BYTES:
        raise ValueError("ByteDance checkpoint size is incorrect. Run: python diagnostics/download_bytedance.py")
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        checksum = digest.hexdigest()
    if checksum != MODEL_SHA256:
        raise ValueError("ByteDance checkpoint checksum does not match. Run: python diagnostics/download_bytedance.py")


def resolve_device(device):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Choose auto, cpu or cuda for ByteDance.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Choose CPU or Auto.")
    return device


@lru_cache(maxsize=1)
def _load_model(device):
    from ._bytedance.models import Note_pedal
    verify_checkpoint()
    saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # All parameters, including frozen frontend kernels, must come from the file.
    # Meta construction avoids initializing 40M weights only to overwrite them.
    with torch.device("meta"):
        model = Note_pedal(frames_per_second=FPS, classes_num=88)
    model.load_state_dict(saved["model"], strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device)


def read_audio(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError("Choose an uncompressed 16-bit PCM WAV file.")
        rate, channels, length = audio.getframerate(), audio.getnchannels(), audio.getnframes()
        if length == 0:
            raise ValueError("The recording is empty.")
        samples = np.frombuffer(audio.readframes(length), dtype="<i2").reshape(-1, channels)
    samples = samples.astype(np.float32).mean(axis=1) / 32768.0
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return samples, length / rate


def segment_plan(sample_count):
    """Upstream pads to whole 10s, calculates at 5s strides, keeps centers."""
    sections = max(1, math.ceil(sample_count / SEGMENT_SAMPLES))
    count = 2 * sections - 1
    for index in range(count):
        # Duplicate FFT endpoints are omitted in the multi-window case.
        first = 0 if index == 0 else 250
        last = 1001 if count == 1 else (1000 if index == count - 1 else 750)
        yield index * STRIDE_SAMPLES, first, last


def predict_probabilities(samples, model, device, *, cancel=None, progress=None):
    limit = len(samples) // (SAMPLE_RATE // FPS) + 1
    outputs = {key: np.zeros((limit, width), dtype=np.float32) for key, width in OUTPUT_WIDTHS.items()}
    plan, cursor = list(segment_plan(len(samples))), 0
    with torch.inference_mode():
        for index, (start, first, last) in enumerate(plan):
            _check_cancel(cancel)
            window = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
            chunk = samples[start:start + SEGMENT_SAMPLES]
            window[:len(chunk)] = chunk
            predictions = model(torch.from_numpy(window).unsqueeze(0).to(device))
            length = min(last - first, limit - cursor)
            for key, width in OUTPUT_WIDTHS.items():
                values = predictions[key][0, first:first + length].detach().cpu().numpy()
                if values.shape != (length, width) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid ByteDance prediction: {key}.")
                outputs[key][cursor:cursor + length] = values
            cursor += length
            _check_cancel(cancel)
            if progress is not None:
                progress(index + 1, len(plan))
    _check_cancel(cancel)
    return {key: value[:cursor] for key, value in outputs.items()}


def decode_probabilities(outputs, duration, cancel=None):
    from ._bytedance.decoder import RegressionPostProcessor
    decoder = RegressionPostProcessor(FPS, 88, THRESHOLDS["onset"], THRESHOLDS["offset"],
                                      THRESHOLDS["frame"], THRESHOLDS["pedal_offset"])
    decoder.cancel = cancel
    _check_cancel(cancel)
    # A quiet tail flushes held notes/pedal at the clip boundary. It is never
    # played; returned events are clamped to the actual recording duration.
    padded = {key: np.pad(value, ((0, 12), (0, 0))) for key, value in outputs.items()}
    events, pedal_events = decoder.output_dict_to_midi_events(padded)
    notes, pedals = [], []
    for event in events:
        _check_cancel(cancel)
        start, end = max(0., float(event["onset_time"])), min(duration, float(event["offset_time"]))
        if end <= start:
            continue
        pitch = int(event["midi_note"])
        first = min(len(outputs["frame_output"]) - 1, max(0, round(start * FPS)))
        last = min(len(outputs["frame_output"]), max(first + 1, round(end * FPS)))
        confidence = float(outputs["frame_output"][first:last, pitch - 21].mean())
        notes.append({"name": pitch_name(pitch), "pitch": pitch, "start": start, "end": end,
                      "velocity": max(1, min(127, int(event["velocity"]))), "confidence": confidence})
    for event in pedal_events or []:
        start, end = max(0., float(event["onset_time"])), min(duration, float(event["offset_time"]))
        if end > start:
            pedals.append({"start": start, "end": end})
    notes.sort(key=lambda note: (note["start"], note["pitch"]))
    return notes, pedals


def chord_spans(notes):
    boundaries = defaultdict(list)
    for note in notes:
        boundaries[note["start"]].append((note["pitch"], 1))
        boundaries[note["end"]].append((note["pitch"], -1))
    active, chords, name, start = Counter(), [], None, 0.0
    for timing, changes in sorted(boundaries.items()):
        for pitch, delta in changes:
            active[pitch] += delta
        next_name = chord_name([pitch for pitch, count in active.items() if count > 0])
        if next_name != name:
            if name and timing - start >= 0.2 - 1e-9:
                chords.append({"name": name, "start": start, "end": timing})
            name, start = next_name, timing
    return chords


def transcribe_bytedance(audio, *, device="auto", progress=None, cancel: threading.Event | None = None):
    started = time.monotonic()
    _check_cancel(cancel)
    samples, duration = read_audio(audio)
    device = resolve_device(device)
    _check_cancel(cancel)
    model = _load_model(device)
    _check_cancel(cancel)
    outputs = predict_probabilities(samples, model, device, progress=progress, cancel=cancel)
    notes, pedals = decode_probabilities(outputs, duration, cancel)
    _check_cancel(cancel)
    return {"audio": str(Path(audio).resolve()), "model": "ByteDance piano transcription",
            "engine": "bytedance", "architecture": "bytedance-note-pedal-crnn", "source_commit": SOURCE_COMMIT,
            "duration": duration, "analysis_seconds": time.monotonic() - started, "device": device,
            "thresholds": {key: THRESHOLDS[key] for key in ("frame", "onset", "offset")},
            "notes": notes, "chords": chord_spans(notes),
            "pedals": pedals, "pedal_detection": True,
            "inference_config": {"engine": "bytedance", "sample_rate": SAMPLE_RATE, "frames_per_second": FPS,
                                 "segment_seconds": 10, "overlap_seconds": 5, "overlap_blend": "crop",
                                 "thresholds": dict(THRESHOLDS), "clip_boundary": "close-and-clamp"},
            "playback_config": {"pitch_bends": False}}
