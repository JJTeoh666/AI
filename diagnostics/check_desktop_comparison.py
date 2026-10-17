"""Exercise both real models and the desktop's independent piano-audio caches."""
import json
import sys
import tempfile
import tkinter as tk
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piano_ml.app import PianoApp
from piano_ml.viewer import read_prediction


def main():
    project = Path(__file__).resolve().parent.parent
    audio = project / "diagnostics/browser-clip.wav"
    model = project / "checkpoints/piano-v8.1.pt"
    root = tk.Tk()
    root.withdraw()
    with patch.object(PianoApp, "refresh_models"), patch("piano_ml.app.playback_available", return_value=False):
        app = PianoApp(root)
    try:
        app.audio.set(str(audio))
        app._worker(audio, model, {}, "cpu", 2, "weighted")
        app.poll()
        assert app.result and app.result_kind == "own", app.status.get()
        own_settings = app.threshold.get(), app.onset_threshold.get(), app.offset_threshold.get(), app.overlap_method.get()
        app._bytedance_worker(audio, "cpu")
        app.poll()
        assert set(app.results) == {"own", "bytedance"}, app.status.get()
        assert own_settings == (app.threshold.get(), app.onset_threshold.get(), app.offset_threshold.get(), app.overlap_method.get())
        report = {"recording": str(audio), "piano": "same installed piano engine for both results", "results": {}}
        with tempfile.TemporaryDirectory(dir=project) as directory:
            for key in ("own", "bytedance"):
                result = app.results[key]
                wav = Path(directory) / f"{key}.wav"
                export = Path(directory) / f"{key}-export.wav"
                app.audio_cancel.clear()
                app.audio_worker(result, wav, export, key)
                app.poll()
                assert app.audio_cache[key] == wav and export.read_bytes() == wav.read_bytes()
                with wave.open(str(wav), "rb") as stream:
                    channels, rate = stream.getnchannels(), stream.getframerate()
                    samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
                assert samples.size and np.max(np.abs(samples.astype(float))) > 0
                path = Path(directory) / f"{key}.json"
                path.write_text(json.dumps(result), encoding="utf-8")
                assert len(read_prediction(path)["notes"]) == len(result["notes"])
                report["results"][key] = {"notes": len(result["notes"]), "chords": len(result["chords"]),
                                           "duration": result["duration"], "pedals": len(result.get("pedals", [])), "analysis_seconds": result["analysis_seconds"],
                                           "channels": channels, "sample_rate": rate,
                                           "pcm_peak": float(np.max(np.abs(samples.astype(float))))}
            assert app.audio_cache["own"] != app.audio_cache["bytedance"]
        report["checks"] = ["same WAV", "real checkpoint and ByteDance inference", "own settings preserved",
                            "separate cached piano WAVs", "non-silent audio", "JSON and audio export"]
        output = project / "diagnostics/desktop-comparison-check.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
