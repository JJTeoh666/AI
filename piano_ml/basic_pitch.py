"""Offline Spotify Basic Pitch comparison with the official ONNX model.

Framing and note decoding adapted from Spotify's inference.py/note_creation.py.
Copyright 2022 Spotify AB. Licensed under the Apache License, Version 2.0.
Local modifications: NumPy/SciPy-only adapter, cancellation and app result schema.
See assets/basic-pitch/LICENSE and ATTRIBUTION.txt.
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
from scipy.signal import argrelmax, resample_poly

from .decode import chord_name, pitch_name

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets/basic-pitch/nmp.onnx"
MODEL_SHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"
SOURCE_COMMIT = "fa5997af0a8210982619003269994a1be25eddf3"
SAMPLE_RATE, FFT_HOP = 22050, 256
WINDOW_SAMPLES, HOP_SAMPLES, PADDING = 43844, 36164, 3840
WINDOW_FRAMES, TRIM_FRAMES = 172, 15
ONSET_THRESHOLD, FRAME_THRESHOLD, MIN_NOTE_FRAMES = 0.5, 0.3, 11


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Basic Pitch analysis cancelled.")


@lru_cache(maxsize=1)
def _session():
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Basic Pitch needs ONNX Runtime. Run: python -m pip install -r requirements.txt") from error
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Basic Pitch model is missing. Run: python diagnostics/download_basic_pitch.py")
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Basic Pitch model checksum does not match. Run: python diagnostics/download_basic_pitch.py")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])


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


def audio_window(samples, index):
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    first = index * HOP_SAMPLES - PADDING
    start, end = max(0, first), min(len(samples), first + WINDOW_SAMPLES)
    if end > start:
        window[start - first:end - first] = samples[start:end]
    return window.reshape(1, WINDOW_SAMPLES, 1)


def frame_time(index):
    # Spotify's correction for the model's window/frame alignment.
    offset = FFT_HOP / SAMPLE_RATE * (WINDOW_FRAMES - WINDOW_SAMPLES / FFT_HOP) + 0.0018
    return index * FFT_HOP / SAMPLE_RATE - offset * (index // WINDOW_FRAMES)


def decode_notes(frames, onsets, cancel=None):
    """Spotify's onset + remaining-energy (Melodia) polyphonic decoder.

    Uses its default thresholds and 11-frame minimum, without pitch bends.
    """
    _check_cancel(cancel)
    if len(frames) == 0:
        return []
    changes = []
    for n in (1, 2):
        padded = np.pad(frames, ((n, 0), (0, 0)))
        changes.append(padded[n:] - padded[:-n])
    differences = np.maximum(0, np.minimum(*changes))
    differences[:2] = 0
    maximum = differences.max()
    if maximum > 0:  # Silent inputs must not divide by zero.
        differences *= onsets.max() / maximum
    inferred = np.maximum(onsets, differences)
    peaks = np.zeros_like(inferred)
    indices = argrelmax(inferred, axis=0)
    peaks[indices] = inferred[indices]
    starts, pitches = np.where(peaks >= ONSET_THRESHOLD)
    remaining = frames.copy()
    events = []
    count = len(frames)

    def erase(start, end, pitch):
        remaining[start:end, max(0, pitch - 1):min(88, pitch + 2)] = 0

    for start, pitch in zip(starts[::-1], pitches[::-1]):
        _check_cancel(cancel)
        if start >= count - 1:
            continue
        index, quiet = start + 1, 0
        while index < count - 1 and quiet < 11:
            quiet = quiet + 1 if remaining[index, pitch] < FRAME_THRESHOLD else 0
            index += 1
        end = index - quiet
        if end - start <= MIN_NOTE_FRAMES:
            continue
        erase(start, end, pitch)
        events.append((int(start), int(end), int(pitch + 21), float(frames[start:end, pitch].mean())))

    while remaining.max() > FRAME_THRESHOLD:
        _check_cancel(cancel)
        middle, pitch = np.unravel_index(remaining.argmax(), remaining.shape)
        remaining[middle, pitch] = 0
        index, quiet = middle + 1, 0
        while index < count - 1 and quiet < 11:
            quiet = quiet + 1 if remaining[index, pitch] < FRAME_THRESHOLD else 0
            erase(index, index + 1, pitch)
            index += 1
        end = index - 1 - quiet
        index, quiet = middle - 1, 0
        while index > 0 and quiet < 11:
            quiet = quiet + 1 if remaining[index, pitch] < FRAME_THRESHOLD else 0
            erase(index, index + 1, pitch)
            index -= 1
        start = index + 1 + quiet
        if end - start > MIN_NOTE_FRAMES:
            events.append((int(start), int(end), int(pitch + 21), float(frames[start:end, pitch].mean())))
    return events


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


def transcribe_basic_pitch(audio, *, progress=None, cancel: threading.Event | None = None):
    """Analyze a WAV locally; no PyTorch checkpoint or server is required."""
    started = time.monotonic()
    _check_cancel(cancel)
    samples, duration = read_audio(audio)
    _check_cancel(cancel)
    session = _session()
    # Follow the official Python adapter's overlap removal and length calculation.
    length = int(len(samples) / HOP_SAMPLES * (WINDOW_FRAMES - 2 * TRIM_FRAMES))
    total = math.ceil((len(samples) + PADDING) / HOP_SAMPLES)
    frames, onsets = [], []
    for index in range(total):
        _check_cancel(cancel)
        note, onset = session.run(["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
                                 {"serving_default_input_2:0": audio_window(samples, index)})
        frames.append(note[0, TRIM_FRAMES:-TRIM_FRAMES])
        onsets.append(onset[0, TRIM_FRAMES:-TRIM_FRAMES])
        if progress is not None:
            progress(index + 1, total)
    _check_cancel(cancel)
    events = decode_notes(np.concatenate(frames)[:length], np.concatenate(onsets)[:length], cancel)
    notes = []
    for first, last, pitch, amplitude in events:
        start, end = max(0.0, frame_time(first)), min(duration, frame_time(last))
        if end > start:
            notes.append({"name": pitch_name(pitch), "pitch": pitch, "start": start, "end": end,
                          "confidence": amplitude, "velocity": max(1, min(127, round(amplitude * 127)))})
    notes.sort(key=lambda note: (note["start"], note["pitch"]))
    return {"audio": str(Path(audio).resolve()), "model": "Spotify Basic Pitch", "engine": "basic-pitch",
            "architecture": "spotify-basic-pitch", "source_commit": SOURCE_COMMIT, "duration": duration,
            "analysis_seconds": time.monotonic() - started, "thresholds": {"frame": .3, "onset": .5},
            "notes": notes, "chords": chord_spans(notes), "pedals": [], "pedal_detection": False,
            "min_note_seconds": MIN_NOTE_FRAMES * FFT_HOP / SAMPLE_RATE,
            "inference_config": {"engine": "basic-pitch", "sample_rate": SAMPLE_RATE,
                                 "window_samples": WINDOW_SAMPLES, "hop_samples": HOP_SAMPLES,
                                 "onset_threshold": .5, "frame_threshold": .3,
                                 "minimum_note_frames": 11, "infer_onsets": True, "melodia_trick": True},
            "playback_config": {"pitch_bends": False}}
