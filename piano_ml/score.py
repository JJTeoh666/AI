"""Approximate grand-staff notation for timed piano detections."""

import math
from functools import lru_cache

import numpy as np
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Arc, Ellipse, PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

DEGREES = (0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6)
SHARPS = (1, 3, 6, 8, 10)


def staff_pitch(pitch: int) -> tuple[str, float, bool]:
    """Return clef, vertical position and sharp spelling (C4 is middle C)."""
    diatonic = (pitch // 12 - 1) * 7 + DEGREES[pitch % 12]
    if pitch >= 60:
        return "treble", (diatonic - 30) / 2, pitch % 12 in SHARPS
    return "bass", -10 + (diatonic - 18) / 2, pitch % 12 in SHARPS


def note_value(seconds: float, bpm: float) -> tuple[float, int, bool]:
    """Nearest conventional duration in quarter-note beats, flags and dot."""
    beats = max(0.01, seconds * bpm / 60)
    values = (4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125)
    value = min(values, key=lambda candidate: abs(math.log(beats / candidate)))
    dotted = value in (3, 1.5, 0.75, 0.375, 0.1875)
    base = value / 1.5 if dotted else value
    flags = max(0, round(-math.log2(base))) if base < 1 else 0
    return value, flags, dotted


@lru_cache(maxsize=1)
def music_font():
    for name in ("Segoe UI Symbol", "Noto Music", "Bravura", "FreeSerif", "Symbola"):
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            font = FT2Font(path)
            if font.get_char_index(0x1D11E) and font.get_char_index(0x1D122):
                return font_manager.FontProperties(fname=path)
        except ValueError:
            continue
    return None


def _clef(axis, glyph, x, bottom, height, fallback):
    font = music_font()
    if font is None:
        axis.text(x, bottom + height / 2, fallback, ha="center", va="center", fontsize=19)
        return
    path = TextPath((0, 0), glyph, size=1, prop=font)
    box = path.get_extents()
    transform = (Affine2D().translate(-box.x0, -box.y0)
                 .scale(0.52 / box.width, height / box.height)
                 .translate(x - 0.26, bottom) + axis.transData)
    axis.add_patch(PathPatch(path, transform=transform, facecolor="#17243b", linewidth=0))


def draw_staff(figure, result: dict, bpm: float = 120, page: int = 0,
               bars_per_page: int = 2, names: bool = False) -> tuple:
    if not 30 <= bpm <= 300:
        raise ValueError("Notation tempo must be between 30 and 300 BPM.")
    beats_per_page = bars_per_page * 4
    seconds_per_page = beats_per_page * 60 / bpm
    pages = max(1, math.ceil(result["duration"] / seconds_per_page))
    page = max(0, min(page, pages - 1))
    start = page * seconds_per_page
    end = min(result["duration"], start + seconds_per_page)
    visible = sorted((note for note in result["notes"]
                      if note["start"] < end and note["end"] > start),
                     key=lambda note: (note["start"], note["pitch"]))
    figure.clear()
    figure.set_facecolor("white")
    axis = figure.add_subplot(111)
    axis.set_facecolor("white")
    axis.set_xlim(-2.05, beats_per_page + 0.35)
    positions = [staff_pitch(note["pitch"])[1] for note in visible]
    axis.set_ylim(min(-13, min(positions, default=-10) - 3.5),
                  max(8, max(positions, default=4) + 3.5))
    for bottom in (0, -10):
        for line in range(5):
            axis.plot([-1.65, beats_per_page + 0.15], [bottom + line] * 2,
                      color="#667085", linewidth=0.8, zorder=0)
        axis.text(-0.4, bottom + 2.8, "4", ha="center", va="center", fontsize=17)
        axis.text(-0.4, bottom + 1.2, "4", ha="center", va="center", fontsize=17)
    _clef(axis, "\U0001d11e", -1.1, -1.5, 7.0, "G")
    _clef(axis, "\U0001d122", -1.1, -9.2, 3.7, "F")
    # Brace and bar lines join the two five-line staves.
    brace = Path([(-1.70, 4.3), (-1.97, 3.5), (-1.70, 1.0), (-1.85, -1.3),
                  (-1.94, -2.5), (-1.70, -2.6), (-1.97, -3.0),
                  (-1.70, -3.4), (-1.94, -3.5), (-1.85, -4.7),
                  (-1.70, -7.0), (-1.97, -9.5), (-1.70, -10.3)],
                 [Path.MOVETO] + [Path.CURVE4] * 12)
    axis.add_patch(PathPatch(brace, fill=False, linewidth=1.7, color="#17243b"))
    for beat in range(0, beats_per_page + 1, 4):
        axis.plot([beat, beat], [-10, 4], color="#8b96a7", linewidth=0.8, zorder=0)
        axis.text(beat + 0.06, 5.1, str(page * bars_per_page + beat // 4 + 1),
                  fontsize=8, color="#667085")
    accidental_state = {}
    occupied = {}
    for note in visible:
        clef, y, sharp = staff_pitch(note["pitch"])
        local_beat = max(0, (note["start"] - start) * bpm / 60)
        beat = round(local_beat * 4) / 4
        x = min(beats_per_page - 0.08, beat + 0.2)
        key = (beat, clef)
        previous_y = occupied.get(key)
        if previous_y is not None and abs(y - previous_y) <= 0.5:
            x += 0.16
        occupied[key] = y
        value, flags, dotted = note_value(note["end"] - note["start"], bpm)
        hollow = value >= 2
        axis.add_patch(Ellipse((x, y), 0.18, 0.57, angle=0,
                              facecolor="white" if hollow else "#17243b",
                              edgecolor="#17243b", linewidth=1.1, zorder=3))
        bottom = 0 if clef == "treble" else -10
        # Ledger lines occur at every whole step outside the staff.
        if y <= bottom - 1:
            for ledger in np.arange(bottom - 1, y - 0.01, -1):
                axis.plot([x - 0.16, x + 0.16], [ledger] * 2, color="#17243b", linewidth=0.8)
        if y >= bottom + 5:
            for ledger in np.arange(bottom + 5, y + 0.01, 1):
                axis.plot([x - 0.16, x + 0.16], [ledger] * 2, color="#17243b", linewidth=0.8)
        acc_key = (int(beat // 4), clef, y)
        previous_sharp = accidental_state.get(acc_key, False)
        if sharp or previous_sharp != sharp:
            axis.text(x - 0.18, y, "♯" if sharp else "♮", ha="right", va="center", fontsize=13)
        accidental_state[acc_key] = sharp
        if value < 4:
            direction = -1 if y >= bottom + 2 else 1
            stem_x = x + 0.09 * direction
            stem_tip = y + 2.8 * direction
            axis.plot([stem_x] * 2, [y, stem_tip], color="#17243b", linewidth=1.0)
            for flag in range(flags):
                tip = stem_tip - direction * flag * 0.42
                flag_path = Path([(stem_x, tip), (stem_x + 0.18, tip - direction * 0.3),
                                  (stem_x + 0.15, tip - direction * 0.9),
                                  (stem_x + 0.07, tip - direction * 1.1)],
                                 [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                axis.add_patch(PathPatch(flag_path, fill=False, linewidth=1.1, color="#17243b"))
        if dotted:
            axis.add_patch(Ellipse((x + 0.22, y + (0.24 if y % 1 == 0 else 0)),
                                  0.035, 0.12, facecolor="#17243b", edgecolor="none"))
        if note["start"] < start:
            axis.add_patch(Arc((x - 0.15, y - 0.55), 0.30, 0.45,
                               theta1=190, theta2=350, color="#667085", linewidth=0.8))
        if names:
            axis.text(x, y - 1.0, note["name"], ha="center", fontsize=7, color="#126b60")
    for chord in result["chords"]:
        if start <= chord["start"] < end:
            axis.text((chord["start"] - start) * bpm / 60 + 0.2, 6.6,
                      chord["name"], fontsize=8, color="#126b60", clip_on=True)
    axis.set_xticks(range(0, beats_per_page + 1))
    axis.set_xticklabels([f"{start + beat * 60 / bpm:.2f}" for beat in range(beats_per_page + 1)], fontsize=8)
    axis.set_xlabel("Time (seconds); note spacing and durations approximated at the selected tempo", fontsize=9, color="#667085")
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="x", length=0)
    figure.suptitle(f"Piano transcription   ·   {bpm:g} BPM   ·   Page {page + 1}/{pages}   ·   {start:.2f}–{end:.2f} s",
                    fontsize=12, color="#17243b")
    figure.subplots_adjust(left=0.03, right=0.98, bottom=0.12, top=0.89)
    return axis, page, pages, (start, start + seconds_per_page)
