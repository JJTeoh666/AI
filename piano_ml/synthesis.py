"""Render detected notes with a sampled grand piano or a basic fallback tone."""

import math
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .soundfont import sampled_piano_available, render_sampled_wav

RELEASE_SECONDS = 0.18


def piano_sound_name() -> str:
    return "Salamander grand piano" if sampled_piano_available() else "Basic piano tone"


def _voice(times: np.ndarray, pitch: int, held_seconds: float, sample_rate: int) -> np.ndarray:
    frequency = 440 * 2 ** ((pitch - 69) / 12)
    tone = np.zeros_like(times, dtype=np.float32)
    # Higher partials decay faster, producing a soft struck-string timbre.
    for harmonic, amplitude in ((1, 1.0), (2, 0.38), (3, 0.18), (4, 0.10), (5, 0.05), (6, 0.025)):
        if harmonic * frequency >= sample_rate / 2:
            break
        decay = np.exp(-times * (0.7 + 0.8 * (harmonic - 1)))
        tone += amplitude * decay * np.sin(2 * np.pi * frequency * harmonic * times)
    attack = np.minimum(1, times / 0.006)
    release = np.clip((times - held_seconds) / RELEASE_SECONDS, 0, 1)
    envelope = attack * np.cos(release * np.pi / 2) ** 2
    return tone * envelope


def render_result_wav(result: dict, output: str | Path, sample_rate: int | None = None,
                      progress: Callable[[int, int], None] | None = None,
                      cancel: threading.Event | None = None, *, engine: str = "auto") -> Path:
    """Preserve note timing and overlap; stream blocks without loading a song into RAM."""
    if engine not in ("auto", "sampled", "basic"):
        raise ValueError("Choose the auto, sampled or basic piano engine.")
    sampled = engine == "sampled" or engine == "auto" and sampled_piano_available()
    if sample_rate is None:
        sample_rate = 44100 if sampled else 22050
    if not isinstance(sample_rate, int) or not 16000 <= sample_rate <= 96000:
        raise ValueError("Synthesis sample rate must be an integer between 16000 and 96000 Hz.")
    duration = float(result["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Result duration must be positive.")
    notes = sorted([dict(note) for note in result["notes"]], key=lambda note: note["start"])
    for note in notes:
        if not (21 <= int(note["pitch"]) <= 108 and
                math.isfinite(note["start"] + note["end"]) and
                0 <= note["start"] < note["end"] <= duration + 0.001):
            raise ValueError("Invalid note pitch or timing.")
        if not 1 <= note.get("velocity", 100) <= 127:
            raise ValueError("Note velocity must be between 1 and 127.")
    pedals = result.get("pedals", [])
    for pedal in pedals:
        if not (math.isfinite(pedal["start"] + pedal["end"]) and
                0 <= pedal["start"] < pedal["end"] <= duration + 0.001):
            raise ValueError("Invalid sustain pedal timing.")
    if sampled:
        return render_sampled_wav(result, output, sample_rate, progress, cancel)
    # Extend sounding durations at key release without changing the displayed score.
    next_start = {}
    for note in reversed(notes):
        for pedal in pedals:
            if pedal["start"] <= note["end"] < pedal["end"]:
                note["end"] = min(duration, pedal["end"])
        note["end"] = min(note["end"], next_start.get(note["pitch"], duration))
        next_start[note["pitch"]] = note["start"]
    tail = RELEASE_SECONDS if notes else 0
    total_frames = round((duration + tail) * sample_rate)
    total_blocks = math.ceil(total_frames / sample_rate)
    endpoints = sorted([(note["start"], 1) for note in notes] +
                       [(note["end"] + RELEASE_SECONDS, -1) for note in notes])
    voices = peak_voices = 0
    for _, change in endpoints:
        voices += change
        peak_voices = max(peak_voices, voices)
    gain = 0.30 / math.sqrt(max(1, peak_voices))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    next_note = 0
    active = []
    try:
        with wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for block, first in enumerate(range(0, total_frames, sample_rate)):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("Result audio rendering cancelled.")
                count = min(sample_rate, total_frames - first)
                start, end = first / sample_rate, (first + count) / sample_rate
                active = [note for note in active if note["end"] + RELEASE_SECONDS > start]
                while next_note < len(notes) and notes[next_note]["start"] < end:
                    note = notes[next_note]
                    if note["end"] + RELEASE_SECONDS > start:
                        active.append(note)
                    next_note += 1
                mixed = np.zeros(count, dtype=np.float32)
                for note in active:
                    lo = max(0, math.ceil(note["start"] * sample_rate) - first)
                    hi = min(count, math.ceil((note["end"] + RELEASE_SECONDS) * sample_rate) - first)
                    if hi > lo:
                        times = (np.arange(first + lo, first + hi) / sample_rate - note["start"]).astype(np.float32)
                        mixed[lo:hi] += _voice(times, int(note["pitch"]), note["end"] - note["start"], sample_rate) * math.sqrt(note.get("velocity", 100) / 100)
                pcm = (np.tanh(mixed * gain) * 32767).astype("<i2")
                wav.writeframes(pcm.tobytes())
                if progress is not None:
                    progress(block + 1, total_blocks)
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Result audio rendering cancelled.")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output
