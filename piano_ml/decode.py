"""Turn frame probabilities into note events and conservative chord names."""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from .thresholds import Thresholds

from .model import HOP_LENGTH, LOW_NOTE, SAMPLE_RATE

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORDS = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
}


class DecodeCache:
    """Reusable decisions for one audio window and many threshold combinations."""

    def __init__(self, probabilities):
        self.probabilities = probabilities
        frame = probabilities["frame"]
        padded = np.pad(frame.astype(np.result_type(frame.dtype, np.float32), copy=False),
                        ((1, 2), (0, 0)), constant_values=-np.inf)
        self.start_strength = np.maximum.reduce([padded[index:index + len(frame)] for index in range(4)])
        self._peaks = {}
        self._gaps = {}

    def peaks(self, head, values):
        key = (head, values)
        if key not in self._peaks:
            output = self.probabilities.get(head)
            thresholds = np.broadcast_to(np.asarray(values), (self.start_strength.shape[1],))
            self._peaks[key] = [find_peaks(np.pad(output[:, column], (1, 1)),
                height=thresholds[column], **({"distance": 2} if head == "onset" else {}))[0] - 1
                for column in range(output.shape[1])] if output is not None else [
                    np.empty(0, dtype=np.int64) for _ in range(self.start_strength.shape[1])]
        return self._peaks[key]

    def gaps(self, values, release_frames):
        key = (values, release_frames)
        if key not in self._gaps:
            inactive = self.probabilities["frame"] < np.asarray(values)
            if len(inactive) < release_frames:
                self._gaps[key] = [np.empty(0, dtype=np.int64) for _ in range(inactive.shape[1])]
                return self._gaps[key]
            confirmed = inactive[release_frames - 1:].copy()
            for shift in range(1, release_frames):
                confirmed &= inactive[release_frames - 1 - shift:len(inactive) - shift]
            self._gaps[key] = [np.flatnonzero(confirmed[:, column]) + release_frames - 1
                              for column in range(inactive.shape[1])]
        return self._gaps[key]


def pitch_name(pitch: int) -> str:
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


def chord_name(pitches: list[int]) -> str | None:
    classes = {pitch % 12 for pitch in pitches}
    if len(classes) not in (3, 4):
        return None
    for root in sorted(classes):
        shape = {(note - root) % 12 for note in classes}
        for quality, intervals in CHORDS.items():
            if shape == set(intervals):
                return f"{NAMES[root]} {quality}"
    return None


def decode(probabilities: np.ndarray, duration: float, threshold: float = 0.5,
           min_note_seconds: float = 0.08, min_chord_seconds: float = 0.20,
           onset_probabilities: np.ndarray | None = None,
           offset_probabilities: np.ndarray | None = None,
           velocity_probabilities: np.ndarray | None = None,
           pedal_probabilities: np.ndarray | None = None,
           include_chords: bool = True,
           frame_threshold: float | None = None,
           onset_threshold: float | None = None,
           offset_threshold: float | None = None,
           release_frames: int = 2, _cache: DecodeCache | None = None,
           include_details: bool = True) -> dict:
    values = Thresholds(frame=threshold, onset=threshold).override(
        frame=frame_threshold, onset=onset_threshold, offset=offset_threshold)
    frame_values, onset_values, offset_values = (values.array(head) for head in ("frame", "onset", "offset"))
    if not np.isfinite(min_note_seconds) or min_note_seconds < 0:
        raise ValueError("Threshold must be between 0 and 1 and minimum duration nonnegative.")
    if not isinstance(release_frames, int) or release_frames < 2:
        raise ValueError("Release persistence must be at least two frames.")
    hop_seconds = HOP_LENGTH / SAMPLE_RATE
    active = median_filter(probabilities >= frame_values, size=(3, 1), mode="nearest") if onset_probabilities is None else None
    cached_peaks = _cache.peaks("onset", values.onset) if _cache is not None and onset_probabilities is not None else None
    cached_offsets = _cache.peaks("offset", values.offset) if cached_peaks is not None else None
    cached_gaps = _cache.gaps(values.frame, release_frames) if cached_peaks is not None else None
    notes = []
    for column in range(probabilities.shape[1]):
        threshold = frame_values[column]
        if onset_probabilities is not None:
            # Peaks split repeated strikes even when frame activity stays high.
            peaks = cached_peaks[column] if cached_peaks is not None else find_peaks(
                np.pad(onset_probabilities[:, column], (1, 1)), height=onset_values[column], distance=2)[0] - 1
            if cached_offsets is not None:
                offsets = cached_offsets[column]
            elif offset_probabilities is not None:
                offsets = set(find_peaks(np.pad(offset_probabilities[:, column], (1, 1)),
                                        height=offset_values[column])[0] - 1)
            else:
                offsets = set()
            for index, first in enumerate(peaks):
                allowed = (_cache.start_strength[first, column] >= threshold if cached_peaks is not None else
                           np.any(probabilities[max(0, first - 1):first + 3, column] >= threshold))
                if not allowed:
                    continue
                limit = int(peaks[index + 1]) if index + 1 < len(peaks) else len(probabilities)
                last = limit
                if cached_peaks is not None:
                    offset_index = np.searchsorted(offsets, first + 1)
                    offset_frame = int(offsets[offset_index]) if offset_index < len(offsets) else limit
                    gaps = cached_gaps[column]
                    gap_index = np.searchsorted(gaps, first + release_frames)
                    gap_frame = int(gaps[gap_index]) if gap_index < len(gaps) else limit
                    # An offset wins if it coincides with silence confirmation,
                    # matching the frame-by-frame decoder's decision order.
                    if gap_frame < limit and gap_frame < offset_frame:
                        last = gap_frame - release_frames + 1
                    elif offset_frame < limit:
                        last = offset_frame
                else:
                    inactive = 0
                    for frame in range(first + 1, limit):
                        if frame in offsets:
                            last = frame
                            break
                        inactive = inactive + 1 if probabilities[frame, column] < threshold else 0
                        if inactive >= release_frames:
                            last = frame - inactive + 1
                            break
                start, end = first * hop_seconds, min(last * hop_seconds, duration)
                if end - start + 1e-9 >= min_note_seconds:
                    note = {"pitch": column + LOW_NOTE, "start": round(start, 3), "end": round(end, 3)}
                    if include_details:
                        note.update(name=pitch_name(column + LOW_NOTE),
                                    confidence=round(float(probabilities[first:last, column].mean()), 3))
                    if include_details and velocity_probabilities is not None:
                        note["velocity"] = int(np.clip(round(127 * float(velocity_probabilities[first, column])), 1, 127))
                    notes.append(note)
            continue
        changes = np.diff(np.pad(active[:, column].astype(np.int8), (1, 1)))
        for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            start = first * hop_seconds
            end = min(last * hop_seconds, duration)
            if end - start >= min_note_seconds:
                pitch = column + LOW_NOTE
                notes.append({"pitch": pitch, "name": pitch_name(pitch),
                              "start": round(start, 3), "end": round(end, 3),
                              "confidence": round(float(probabilities[first:last, column].mean()), 3)})
    notes.sort(key=lambda item: (item["start"], item["pitch"]))
    # Derive chords from the cleaned note events, not from single noisy frames.
    chords = []
    current = None
    segment_start = 0
    for frame in range(len(probabilities) + 1) if include_chords else ():
        time = frame * hop_seconds
        pitches = [item["pitch"] for item in notes if item["start"] <= time < item["end"]] if frame < len(probabilities) else []
        name = chord_name(pitches)
        if name != current:
            if current is not None:
                start = segment_start * hop_seconds
                end = min(time, duration)
                if end - start >= min_chord_seconds:
                    chords.append({"name": current, "start": round(start, 3), "end": round(end, 3)})
            current = name
            segment_start = frame
    result = {"notes": notes, "chords": chords}
    if include_details and pedal_probabilities is not None:
        down = median_filter(pedal_probabilities[:, 0] >= 0.5, size=3, mode="nearest")
        changes = np.diff(np.pad(down.astype(np.int8), (1, 1)))
        result["pedals"] = [{"start": round(first * hop_seconds, 3),
                             "end": round(min(last * hop_seconds, duration), 3)}
                            for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
                            if min(last * hop_seconds, duration) - first * hop_seconds >= 0.06]
    return result


def decode_outputs(probabilities: dict[str, np.ndarray], duration: float,
                   threshold: float, min_note_seconds: float | None = None,
                   include_chords: bool = True, frame_threshold: float | None = None,
                   onset_threshold: float | None = None, offset_threshold: float | None = None,
                   release_frames: int = 2, _cache: DecodeCache | None = None,
                   include_details: bool = True) -> dict:
    return decode(probabilities["frame"], duration, threshold,
                  min_note_seconds=(0.04 if "onset" in probabilities else 0.08)
                  if min_note_seconds is None else min_note_seconds,
                  include_chords=include_chords,
                  frame_threshold=frame_threshold, onset_threshold=onset_threshold,
                  offset_threshold=offset_threshold,
                  release_frames=release_frames,
                  _cache=_cache, include_details=include_details,
                  **{f"{key}_probabilities": value for key, value in probabilities.items() if key != "frame"})
