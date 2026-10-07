"""Audio windows and aligned frame labels for MAESTRO."""

import csv
import math
import random
import wave
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .midi import read_performance
from .model import HOP_LENGTH, LOW_NOTE, N_NOTES, SAMPLE_RATE
from .middle_data import read_training_exclusions, FOCUS_LOW_NOTE, FOCUS_HIGH_NOTE


def read_wav_window(path: str | Path, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Read 16-bit PCM WAV as mono float32 at 16 kHz."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed 16-bit PCM WAV: {path}")
        source_rate = wav.getframerate()
        wav.setpos(min(int(start * source_rate), wav.getnframes()))
        count = wav.getnframes() if duration is None else int(round(duration * source_rate))
        raw = wav.readframes(count)
        channels = wav.getnchannels()
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32)
    mono = samples.mean(axis=1) / 32768.0
    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, source_rate // divisor).astype(np.float32)
    if duration is not None:
        target = round(duration * SAMPLE_RATE)
        mono = np.pad(mono[:target], (0, max(0, target - len(mono))))
    return mono


def load_records(root: str | Path, split: str) -> list[dict]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}; run the download command first")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == split]
    if split == "train":
        excluded = read_training_exclusions(root)
        records = [row for row in records if row["audio_filename"] not in excluded]
    available = [r for r in records if (root / r["audio_filename"]).exists()
                 and (root / r["midi_filename"]).exists()]
    if not available:
        raise ValueError(f"No complete {split} audio/MIDI pairs found in {root}")
    return available


class MaestroWindows(Dataset):
    def __init__(self, root: str | Path, split: str, seconds: float = 4.0,
                 windows_per_file: int = 8, random_windows: bool = False,
                 max_files: int | None = None, multi_target: bool = False,
                 augment: bool = False, bass_sampling: float = 0.0,
                 bass_max_note: int = 47, treble_sampling: float = 0.0,
                 treble_min_note: int = 84, middle_sampling: float = 0.0,
                 middle_min_note: int = FOCUS_LOW_NOTE, middle_max_note: int = FOCUS_HIGH_NOTE) -> None:
        if seconds <= 0 or windows_per_file < 1:
            raise ValueError("Window duration and windows per file must be positive.")
        self.root = Path(root)
        self.records = load_records(root, split)
        if max_files is not None:
            self.records = self.records[:max_files]
        self.seconds = seconds
        self.windows_per_file = windows_per_file
        self.random_windows = random_windows
        self.multi_target = multi_target
        self.augment = augment
        self._performances: dict[str, tuple] = {}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (bass_sampling, treble_sampling, middle_sampling)):
            raise ValueError("Pitch sampling fractions must be between zero and one.")
        if bass_sampling + treble_sampling + middle_sampling > 1:
            raise ValueError("Pitch sampling fractions must sum to at most one.")
        if not LOW_NOTE <= bass_max_note < LOW_NOTE + N_NOTES:
            raise ValueError("Bass maximum pitch must be a piano MIDI note (21–108).")
        if not LOW_NOTE <= treble_min_note < LOW_NOTE + N_NOTES:
            raise ValueError("Treble minimum pitch must be a piano MIDI note (21–108).")
        if bass_sampling and treble_sampling and bass_max_note >= treble_min_note:
            raise ValueError("Bass and treble sampling ranges must not overlap.")
        if not LOW_NOTE <= middle_min_note <= middle_max_note < LOW_NOTE + N_NOTES:
            raise ValueError("Middle focus must be an inclusive piano MIDI range (21-108).")
        if (bass_sampling or treble_sampling or middle_sampling) and (split != "train" or not random_windows):
            raise ValueError("Pitch-focused sampling is only available for random training windows.")
        self.bass_sampling = bass_sampling
        self.treble_sampling = treble_sampling
        self.middle_sampling = middle_sampling
        self.middle_min_note, self.middle_max_note = middle_min_note, middle_max_note
        self.bass_anchors = []
        self.bass_cumulative = []
        self.treble_anchors = []
        self.treble_cumulative = []
        self.middle_anchors, self.middle_cumulative = [], []
        if bass_sampling or treble_sampling or middle_sampling:
            for row in self.records:
                performance = read_performance(self.root / row["midi_filename"])
                self._performances[row["midi_filename"]] = performance
                for note in performance[0]:
                    if not 0 <= note.start < float(row["duration"]):
                        continue
                    if bass_sampling and LOW_NOTE <= note.pitch <= bass_max_note:
                        self.bass_anchors.append((row, note))
                    if treble_sampling and treble_min_note <= note.pitch < LOW_NOTE + N_NOTES:
                        self.treble_anchors.append((row, note))
                    if middle_sampling and middle_min_note <= note.pitch <= middle_max_note:
                        self.middle_anchors.append((row, note))
            for anchors, cumulative in ((self.bass_anchors, self.bass_cumulative),
                                         (self.treble_anchors, self.treble_cumulative),
                                         (self.middle_anchors, self.middle_cumulative)):
                counts = Counter(note.pitch for _, note in anchors)
                total = 0.0
                for _, note in anchors:
                    total += 1 / math.sqrt(counts[note.pitch])
                    cumulative.append(total)

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_file

    def __getitem__(self, index: int):
        row = self.records[index // self.windows_per_file]
        anchor = None
        if self.bass_sampling or self.treble_sampling or self.middle_sampling:
            draw = random.random()
            if draw < self.bass_sampling and self.bass_anchors:
                row, anchor = random.choices(self.bass_anchors, cum_weights=self.bass_cumulative, k=1)[0]
            elif self.bass_sampling <= draw < self.bass_sampling + self.treble_sampling and self.treble_anchors:
                row, anchor = random.choices(self.treble_anchors, cum_weights=self.treble_cumulative, k=1)[0]
            elif self.bass_sampling + self.treble_sampling <= draw < self.bass_sampling + self.treble_sampling + self.middle_sampling and self.middle_anchors:
                row, anchor = random.choices(self.middle_anchors, cum_weights=self.middle_cumulative, k=1)[0]
        duration = float(row["duration"])
        latest_start = max(0.0, duration - self.seconds)
        slot = index % self.windows_per_file
        if self.random_windows:
            if anchor is None:
                start = random.uniform(0, latest_start)
            else:
                # Include the full note when it fits, with context on both sides.
                earliest = max(0.0, min(anchor.end, anchor.start + self.seconds - 0.2) - self.seconds + 0.1)
                latest = min(latest_start, max(0.0, anchor.start - 0.1))
                start = random.uniform(min(earliest, latest), latest)
        else:
            start = latest_start * (slot + 0.5) / self.windows_per_file
        wave_data = read_wav_window(self.root / row["audio_filename"], start, self.seconds)
        if self.augment:
            wave_data = augment_audio(wave_data)
        midi_name = row["midi_filename"]
        if midi_name not in self._performances:
            self._performances[midi_name] = read_performance(self.root / midi_name)
        notes, pedals = self._performances[midi_name]
        n_frames = len(wave_data) // HOP_LENGTH + 1
        frame_times = start + np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE
        labels = np.zeros((n_frames, N_NOTES), dtype=np.float32)
        targets = {key: np.zeros_like(labels) for key in ("onset", "offset", "velocity")}
        reference = []
        for note in notes:
            if note.end < start or note.start > start + self.seconds:
                continue
            if LOW_NOTE <= note.pitch < LOW_NOTE + N_NOTES:
                labels[:, note.pitch - LOW_NOTE] = np.maximum(
                    labels[:, note.pitch - LOW_NOTE],
                    (frame_times >= note.start) & (frame_times < note.end),
                )
                column = note.pitch - LOW_NOTE
                for key, time in (("onset", note.start), ("offset", note.end)):
                    if start <= time < start + self.seconds:
                        first = min(n_frames - 1, round((time - start) * SAMPLE_RATE / HOP_LENGTH))
                        targets[key][first:first + 2, column] = 1
                        if key == "onset":
                            targets["velocity"][first:first + 2, column] = note.velocity / 127
                reference.append({"pitch": note.pitch, "start": note.start - start,
                                  "end": note.end - start})
        if not self.multi_target:
            return wave_data, labels
        targets["frame"] = labels
        targets["pedal"] = np.zeros((n_frames, 1), np.float32)
        for pedal in pedals:
            targets["pedal"][:, 0] = np.maximum(targets["pedal"][:, 0],
                (frame_times >= pedal.start) & (frame_times < pedal.end))
        targets["reference_notes"] = reference
        return wave_data, targets


def collate_windows(batch):
    """Stack targets while retaining variable-length reference note lists."""
    waves, targets = zip(*batch)
    return default_collate(waves), {
        key: [target[key] for target in targets] if key == "reference_notes"
        else default_collate([target[key] for target in targets]) for key in targets[0]}


def augment_audio(audio: np.ndarray) -> np.ndarray:
    """Small gain, noise and delayed room reflections; pitch and timing stay fixed."""
    output = audio.copy()
    if random.random() < 0.4:
        delay = random.randint(round(0.015 * SAMPLE_RATE), round(0.060 * SAMPLE_RATE))
        for multiple, decay in ((1, 0.10), (2, 0.05)):
            shift = delay * multiple
            if shift < len(output):
                output[shift:] += decay * audio[:-shift]
    rms = float(np.sqrt(np.mean(output ** 2)))
    if rms > 1e-6 and random.random() < 0.5:
        noise = np.random.normal(0, rms * 10 ** (-random.uniform(25, 40) / 20), len(output))
        output += noise.astype(np.float32)
    output *= 10 ** (random.uniform(-6, 3) / 20)
    return np.clip(output, -1, 1).astype(np.float32)
