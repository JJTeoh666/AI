"""Compare overlap methods on fixed validation excerpts without changing checkpoints."""

import argparse
import hashlib
import json
import sys
import tempfile
import time
import wave
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

from piano_ml.data import load_records
from piano_ml.decode import decode_outputs
from piano_ml.inference import get_device, load_model, predict_probabilities
from piano_ml.metrics import note_counts, note_counts_by_pitch, pitch_note_report, precision_recall_f1, window_note_pairs
from piano_ml.midi import read_notes
from piano_ml.model import LOW_NOTE, N_NOTES, SAMPLE_RATE, model_config


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    stat = path.stat()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def summarize(pitch_counts, onset_counts, inference_seconds, analysis_seconds):
    pitch_report = pitch_note_report(pitch_counts)
    note = precision_recall_f1(*pitch_counts.sum(axis=0))
    return {"note_macro_f0123": pitch_report["note_macro_f0123"],
            "note_macro_pitch_count": pitch_report["note_macro_pitch_count"],
            "onset_f1": precision_recall_f1(*onset_counts)["f1"],
            "note_with_offsets_f1": note["f1"], "note_precision": note["precision"],
            "note_recall": note["recall"], "reference_notes": int((pitch_counts[:, 0] + pitch_counts[:, 2]).sum()),
            "predicted_notes": int((pitch_counts[:, 0] + pitch_counts[:, 1]).sum()),
            "inference_seconds": inference_seconds, "analysis_seconds": analysis_seconds,
            "per_pitch_notes": pitch_report["per_pitch_notes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "maestro")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints" / "piano-v8.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "diagnostics" / "overlap-blending-comparison.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = get_device(args.device)
    before = fingerprint(args.checkpoint)
    model = load_model(args.checkpoint, device)
    thresholds = model.decoding_thresholds
    config = model_config(model)
    records = sorted(load_records(args.data, "validation"), key=lambda row: row["audio_filename"])[:3]
    if len(records) != 3:
        raise ValueError("This comparison needs three complete official validation pairs.")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Exclude one shared warmup from timing; each measured method uses identical chunk counts.
    with torch.inference_mode():
        model(torch.zeros((1, 6 * SAMPLE_RATE), device=device))
    synchronize()
    blend_methods = ("crop", "weighted", "equal")
    totals = {method: {"pitch": np.zeros((N_NOTES, 3), dtype=np.int64),
                       "onset": np.zeros(3, dtype=np.int64), "inference_seconds": 0.0, "analysis_seconds": 0.0}
              for method in blend_methods}
    excerpts = []
    with tempfile.TemporaryDirectory(prefix="overlap-comparison-", dir=PROJECT) as directory:
        for index, row in enumerate(records):
            clip = Path(directory) / f"validation-{index}.wav"
            with wave.open(str(args.data / row["audio_filename"]), "rb") as source:
                params = source.getparams()
                frames = min(source.getnframes(), 60 * source.getframerate())
                duration = frames / source.getframerate()
                raw = source.readframes(frames)
            with wave.open(str(clip), "wb") as target:
                target.setparams(params)
                target.writeframes(raw)
            reference = [asdict(note) for note in read_notes(args.data / row["midi_filename"])
                         if LOW_NOTE <= note.pitch < LOW_NOTE + N_NOTES and note.start < duration]
            excerpt = {"audio_filename": row["audio_filename"], "midi_filename": row["midi_filename"],
                       "start_seconds": 0, "duration_seconds": duration, "methods": {}}
            # Rotate so each method occupies each timing position once.
            methods = blend_methods[index:] + blend_methods[:index]
            for method in methods:
                print(f"Validation {index + 1}/3: {method}, {row['audio_filename']}", flush=True)
                synchronize()
                started = time.perf_counter()
                _, probabilities = predict_probabilities(clip, model, device, context_seconds=1,
                                                        overlap_blend=method)
                synchronize()
                inference_seconds = time.perf_counter() - started
                events = decode_outputs(probabilities, duration, thresholds.summary()["frame"], model.min_note_seconds,
                                        frame_threshold=thresholds.frame, onset_threshold=thresholds.onset,
                                        offset_threshold=thresholds.offset, release_frames=config.get("release_frames", 2))
                analysis_seconds = time.perf_counter() - started
                truth, guesses = window_note_pairs(reference, events["notes"], duration)
                pitch_counts = np.zeros((N_NOTES, 3), dtype=np.int64)
                for pitch, counts in note_counts_by_pitch(truth, guesses, offsets=True).items():
                    pitch_counts[pitch - LOW_NOTE] = counts
                onset_counts = np.array(note_counts(truth, guesses, offsets=False), dtype=np.int64)
                excerpt["methods"][method] = summarize(pitch_counts, onset_counts, inference_seconds, analysis_seconds)
                totals[method]["pitch"] += pitch_counts
                totals[method]["onset"] += onset_counts
                totals[method]["inference_seconds"] += inference_seconds
                totals[method]["analysis_seconds"] += analysis_seconds
            excerpts.append(excerpt)
    aggregate = {method: summarize(values["pitch"], values["onset"], values["inference_seconds"],
                                   values["analysis_seconds"]) for method, values in totals.items()}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "checkpoint": before,
              "checkpoint_unchanged": before == fingerprint(args.checkpoint), "architecture": model.architecture,
              "device": str(device), "torch_version": str(torch.__version__), "model_config": config,
              "thresholds": thresholds.to_dict(), "min_note_seconds": model.min_note_seconds,
              "inference_config": {"chunk_seconds": 4, "context_seconds": 1, "overlap_seconds": 2},
              "selection": "First three complete official validation pairs sorted by audio_filename; first 60 seconds",
              "matching": {"clip_boundary_margin_seconds": 0.1, "onset_tolerance_seconds": 0.05,
                           "offset_tolerance": "max(0.05 seconds, 20% of reference duration)"},
              "timing": "One warmup excluded; inference includes WAV reading and resampling; analysis also includes decoding",
              "excerpts": excerpts, "aggregate": aggregate,
              "weighted_minus_crop": {key: aggregate["weighted"][key] - aggregate["crop"][key]
                                      for key in ("note_macro_f0123", "onset_f1", "note_with_offsets_f1", "analysis_seconds")},
              "equal_minus_crop": {key: aggregate["equal"][key] - aggregate["crop"][key]
                                   for key in ("note_macro_f0123", "onset_f1", "note_with_offsets_f1", "analysis_seconds")},
              "equal_minus_weighted": {key: aggregate["equal"][key] - aggregate["weighted"][key]
                                       for key in ("note_macro_f0123", "onset_f1", "note_with_offsets_f1", "analysis_seconds")},
              "limitation": "Three short validation excerpts are a spot check, not full validation or threshold calibration."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({method: {key: value for key, value in values.items() if key != "per_pitch_notes"}
                      for method, values in aggregate.items()}, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
