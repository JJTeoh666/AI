"""Window and complete-performance evaluation, with validation calibration."""

import json
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MaestroWindows, collate_windows, load_records
from .calibration import Calibration
from .inference import get_device, load_model, predict_probabilities
from .thresholds import requested_grid, Thresholds
from .midi import read_notes
from .model import (HOP_LENGTH, SAMPLE_RATE, LOW_NOTE, N_NOTES, model_config,
                    default_selection_metric, checkpoint_format_version)
from .training import save_checkpoint, score
from .avgf import avgf_accumulators, event_targets


def evaluate(args):
    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    grid = requested_grid(args, model.decoding_thresholds, model.architecture)
    if args.split != "validation" and (len(grid) > 1 or args.calibrate_output):
        raise ValueError("Threshold calibration requires --split validation. Keep test data for final evaluation.")
    metric = args.selection_metric
    if metric == "auto":
        metric = default_selection_metric(model.architecture, getattr(model, "model_version", None))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("training_config", {})
    avgf_options = {"temperature": config.get("threshold_temperature", 0.1),
                    "regularization": config.get("threshold_regularization", 0.05)}
    if metric == "avgf_loss" and model.architecture != "onsets-fourier-recurrent":
        raise ValueError("AvgF loss selection requires a V8/V8.1 Fourier model.")
    minimum = args.min_note_seconds if args.min_note_seconds is not None else model.min_note_seconds
    if args.full_recordings:
        root = Path(args.data)
        records = load_records(root, args.split)
        if args.max_files is not None:
            records = records[:args.max_files]
        calibration = Calibration(grid, minimum, release_frames=model_config(model).get("release_frames", 2))
        avgf = avgf_accumulators(model, grid, **avgf_options)
        for index, record in enumerate(records, 1):
            print(f"Evaluating recording {index}/{len(records)}: {record['audio_filename']}", flush=True)
            duration, probabilities = predict_probabilities(root / record["audio_filename"], model, device)
            reference = [asdict(note) for note in read_notes(root / record["midi_filename"])
                         if LOW_NOTE <= note.pitch < LOW_NOTE + N_NOTES and note.start < duration]
            for note in reference:
                note["end"] = min(note["end"], duration)
            truth = np.zeros_like(probabilities["frame"], dtype=bool)
            for note in reference:
                first = max(0, int(np.ceil(note["start"] * SAMPLE_RATE / HOP_LENGTH)))
                last = min(len(truth), int(np.ceil(note["end"] * SAMPLE_RATE / HOP_LENGTH)))
                truth[first:last, note["pitch"] - LOW_NOTE] = True
            calibration.add(probabilities, truth, reference, duration)
            if avgf:
                targets = event_targets(truth, reference, duration)
                for accumulator in avgf.values():
                    accumulator.add(probabilities, targets)
        result = {**calibration.results(metric, {values: accumulator.results() for values, accumulator in avgf.items()}),
                  "recordings": len(records), "mode": "full_recordings"}
    else:
        data = MaestroWindows(args.data, args.split, seconds=args.seconds, max_files=args.max_files,
                              windows_per_file=args.windows_per_file, multi_target=True)
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_windows)
        result = {**score(model, loader, device, selection_metric=metric,
                         min_note_seconds=minimum, threshold_configs=grid,
                         avgf_temperature=avgf_options["temperature"], avgf_regularization=avgf_options["regularization"]),
                  "recordings": len(data.records), "windows": len(data), "mode": "interior_windows"}
    result.update(split=args.split, architecture=model.architecture, selection_metric=metric,
                  selection_mode="min" if metric == "avgf_loss" else "max",
                  min_note_seconds=minimum, onset_tolerance_seconds=0.05,
                  offset_tolerance="max(0.05 seconds, 20% of reference duration)")
    if hasattr(model, "model_version"):
        result["model_version"] = model.model_version
    chosen = Thresholds(**result["thresholds"])
    if chosen.pitch_dependent:
        result["register_thresholds"] = chosen.registers()
    if args.calibrate_output:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if hasattr(model, "model_version"):
            payload["model_version"] = model.model_version
        payload.update(format_version=4, threshold=result["threshold"], thresholds=result["thresholds"], min_note_seconds=minimum,
                       calibration={"split": args.split, "metric": metric, "mode": result["mode"],
                                    "threshold_curve": result["threshold_curve"]})
        if hasattr(model, "threshold_module"):
            if any(np.any((chosen.array(head) <= 0.05) | (chosen.array(head) >= 0.95))
                   for head in ("frame", "onset", "offset")):
                raise ValueError("Saved learned thresholds must be inside the model's bounds (0.05, 0.95).")
            model.threshold_module.initialize(chosen)
            payload.update(format_version=checkpoint_format_version(model.architecture),
                           model=model.cpu().state_dict())
            # New decisions invalidate the old optimizer momentum and comparison.
            for key in ("optimizer", "scheduler", "validation_signature", "best_score", "best_f1", "stale_epochs", "validation"):
                payload.pop(key, None)
        output = Path(args.calibrate_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(payload, output)
        result["calibrated_checkpoint"] = str(output)
    content = json.dumps(result, indent=2)
    print(content)
    if args.output:
        Path(args.output).write_text(content + "\n", encoding="utf-8")
    pitch_output = getattr(args, "pitch_output", None)
    if pitch_output is None and args.output:
        path = Path(args.output)
        pitch_output = path.with_name(path.stem + ".pitches.csv")
    if pitch_output:
        pitch_output = Path(pitch_output)
        pitch_output.parent.mkdir(parents=True, exist_ok=True)
        with pitch_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("pitch", "name", "reference_count", "predicted_count",
                "tp", "fp", "fn", "precision", "recall", "f0", "f1", "f2", "f3", "f03", "f0123",
                "in_macro_f1", "in_macro_f03", "in_macro_f0123"))
            writer.writeheader()
            writer.writerows(result["per_pitch_notes"])
        print(f"Per-pitch note report: {pitch_output}", flush=True)
    return result
