"""Plot and validate transcription results independently of the GUI."""

import json
import math
import wave
from pathlib import Path

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .decode import pitch_name
from .plot_fonts import plot_font_families


def read_prediction(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    duration = float(result["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The JSON must contain a positive audio duration.")
    for kind in ("notes", "chords"):
        if not isinstance(result.get(kind), list):
            raise ValueError(f"The JSON must contain a {kind} list.")
        for event in result[kind]:
            start, end = float(event["start"]), float(event["end"])
            if not math.isfinite(start + end) or not 0 <= start < end <= duration + 0.001:
                raise ValueError(f"Invalid {kind} timing in JSON.")
            if kind == "notes":
                pitch = int(event["pitch"])
                confidence = float(event.get("confidence", 1))
                if not 21 <= pitch <= 108 or not 0 <= confidence <= 1:
                    raise ValueError("Invalid note pitch or confidence in JSON.")
                event["pitch"] = pitch
                event["confidence"] = confidence
                event["name"] = pitch_name(pitch)
                if "velocity" in event:
                    velocity = int(event["velocity"])
                    if not 1 <= velocity <= 127:
                        raise ValueError("Invalid note velocity in JSON.")
                    event["velocity"] = velocity
            else:
                event["name"] = str(event["name"])
            event["start"], event["end"] = start, end
    result["duration"] = duration
    if "pedals" in result:
        if not isinstance(result["pedals"], list):
            raise ValueError("Pedals must be a list in JSON.")
        for pedal in result["pedals"]:
            start, end = float(pedal["start"]), float(pedal["end"])
            if not math.isfinite(start + end) or not 0 <= start < end <= duration + 0.001:
                raise ValueError("Invalid pedal timing in JSON.")
            pedal["start"], pedal["end"] = start, end
    return result


def waveform_envelope(path: str | Path, points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Waveform preview requires 16-bit PCM WAV.")
        rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
        block = max(1, math.ceil(frames / points))
        times, peaks = [], []
        for start in range(0, frames, block):
            samples = np.frombuffer(wav.readframes(block), dtype="<i2")
            if len(samples):
                mono = samples.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
                times.append((start + len(mono) / 2) / rate)
                peaks.append(float(np.max(np.abs(mono))))
    return np.asarray(times), np.asarray(peaks)


def active_at(result: dict, seconds: float) -> tuple[list[str], list[str]]:
    notes = [note["name"] for note in result["notes"] if note["start"] <= seconds < note["end"]]
    chords = [chord["name"] for chord in result["chords"] if chord["start"] <= seconds < chord["end"]]
    return notes, chords


def draw_prediction(figure: Figure, result: dict,
                    envelope: tuple[np.ndarray, np.ndarray] | None = None) -> list:
    figure.clear()
    figure.set_facecolor("#ffffff")
    audio_name = Path(result.get("audio") or "Piano recording").name
    model_name = Path(result.get("model") or "Saved result").name
    figure.suptitle(f"{audio_name}  |  {model_name}  |  threshold {result.get('threshold', 0.5):.2f}",
                    x=0.075, ha="left", fontsize=10, color="#344054",
                    fontfamily=plot_font_families())
    grid = figure.add_gridspec(3, 1, height_ratios=(1, 4, 0.9), hspace=0.08)
    wave_axis = figure.add_subplot(grid[0])
    note_axis = figure.add_subplot(grid[1], sharex=wave_axis)
    chord_axis = figure.add_subplot(grid[2], sharex=wave_axis)
    if envelope is not None:
        times, peaks = envelope
        wave_axis.fill_between(times, -peaks, peaks, color="#526b83", linewidth=0)
        wave_axis.set_ylim(-1, 1)
    else:
        wave_axis.text(0.5, 0.5, "Original WAV unavailable; detected notes are shown below",
                       transform=wave_axis.transAxes, ha="center", color="#667085")
    wave_axis.set_ylabel("Audio")
    wave_axis.set_yticks([])
    patches, confidence = [], []
    for note in result["notes"]:
        patches.append(Rectangle((note["start"], note["pitch"] - 0.4),
                                 note["end"] - note["start"], 0.8))
        confidence.append(note.get("confidence", 1))
    if patches:
        collection = PatchCollection(patches, cmap="viridis", edgecolors="none")
        collection.set_array(np.asarray(confidence))
        collection.set_clim(0, 1)
        note_axis.add_collection(collection)
        colorbar_axis = figure.add_axes([0.925, 0.25, 0.015, 0.5])
        colorbar = figure.colorbar(collection, cax=colorbar_axis)
        colorbar.set_label("Confidence")
    for pitch in range(21, 109):
        if pitch % 12 in (1, 3, 6, 8, 10):
            note_axis.axhspan(pitch - 0.5, pitch + 0.5, color="#f3f5f8", zorder=0)
    note_axis.set_ylim(20.5, 108.5)
    ticks = [24, 36, 48, 60, 72, 84, 96, 108]
    note_axis.set_yticks(ticks, [pitch_name(pitch) for pitch in ticks])
    note_axis.set_ylabel("Piano key")
    note_axis.grid(axis="y", color="#dfe5eb", linewidth=0.5)
    label_ends = [-math.inf] * 3
    for index, chord in enumerate(result["chords"]):
        chord_axis.axvspan(chord["start"], chord["end"],
                          color=("#d8f0e8", "#e2eafa")[index % 2], alpha=0.95)
        center = (chord["start"] + chord["end"]) / 2
        half_width = max(4, len(chord["name"])) * result["duration"] * 0.0032
        for row in range(3):
            if center - half_width >= label_ends[row]:
                chord_axis.text(center, (0.2, 0.5, 0.8)[row], chord["name"],
                                ha="center", va="center", fontsize=8, clip_on=True,
                                fontfamily=plot_font_families())
                label_ends[row] = center + half_width
                break
    if not result["chords"]:
        chord_axis.text(0.5, 0.5, "No matching chord spans detected", ha="center",
                        transform=chord_axis.transAxes, color="#667085")
    chord_axis.set_ylim(0, 1)
    chord_axis.set_yticks([])
    chord_axis.set_ylabel("Chords")
    chord_axis.set_xlabel("Time (seconds)")
    chord_axis.set_xlim(0, result["duration"])
    for axis in (wave_axis, note_axis, chord_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#475467")
    wave_axis.tick_params(labelbottom=False)
    note_axis.tick_params(labelbottom=False)
    figure.subplots_adjust(left=0.075, bottom=0.08, top=0.92, right=0.89)
    return [wave_axis, note_axis, chord_axis]
