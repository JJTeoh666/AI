"""Measure desktop cursor updates with a dense synthetic transcription."""

import argparse
import json
from pathlib import Path
import time
import tkinter as tk
from unittest.mock import patch

import numpy as np

from piano_ml.app import PianoApp
from piano_ml.decode import pitch_name


def dense_result(count=3000):
    duration = 120.0
    spacing = duration / count
    return {"duration": duration, "model": "Dense playback benchmark", "threshold": .5,
            "notes": [{"name": pitch_name(21 + index % 88), "pitch": 21 + index % 88,
                       "start": index * spacing, "end": min(duration, index * spacing + .3),
                       "confidence": .85, "velocity": 85} for index in range(count)],
            "chords": [{"name": "C major", "start": index * .4, "end": index * .4 + .3}
                       for index in range(300)], "pedals": []}


def measure():
    root = tk.Tk()
    root.attributes("-alpha", 0)
    root.geometry("1260x850")
    with patch.object(PianoApp, "refresh_models"), patch("piano_ml.app.playback_available", return_value=False):
        app = PianoApp(root)
    try:
        result = dense_result()
        app.show_result(result)
        root.update()
        timings = {}
        for name, tab in (("staff", app.staff_tab), ("timeline", app.tabs.tabs()[1]),
                          ("notes", app.tabs.tabs()[2])):
            app.tabs.select(tab)
            root.update()
            samples = []
            for index in range(30):
                start = time.perf_counter()
                app.update_position(.05 + index * .05)
                root.update_idletasks()
                samples.append((time.perf_counter() - start) * 1000)
            timings[name] = {"median_ms": round(float(np.median(samples)), 3),
                             "p95_ms": round(float(np.percentile(samples, 95)), 3)}
        start = time.perf_counter()
        app.tabs.select(app.staff_tab)
        app.update_position(4.1)
        root.update()
        timings["staff_page_change_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return {"notes": len(result["notes"]), "chords": len(result["chords"]),
                "cursor_update": timings}
    finally:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
