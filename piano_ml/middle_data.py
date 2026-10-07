"""Refresh active training data with middle-rich MAESTRO pairs, retaining old files."""

import csv
import json
import math
import shutil
import time
import uuid
import zipfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .bass_data import MIDI_SHA256, _midi_archive
from .download import BASE, CSV_NAME, _member_name, remote_zip_class
from .midi import read_performance_bytes

EXCLUSIONS_NAME = "training-exclusions.json"
FOCUS_LOW_NOTE = 30
FOCUS_HIGH_NOTE = 68


def read_training_exclusions(root):
    path = Path(root) / EXCLUSIONS_NAME
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("excluded_audio_filenames", [])
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("Training exclusions must contain a list of audio filenames.")
    return set(names)


def _safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Training data path escapes its root: {relative}")
    return path


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_middle_refresh(existing, candidates, budget, low_note=FOCUS_LOW_NOTE, high_note=FOCUS_HIGH_NOTE):
    """Retire lower middle-note fractions, then cover scarce middle keys per byte."""
    retired, remaining = [], budget
    for row in sorted(existing, key=lambda row: (row["middle_fraction"], row["middle_count"] / row["bytes"],
                                                 row["audio_filename"])):
        if row["bytes"] <= remaining and len(retired) < len(existing) - 1:
            retired.append(row)
            remaining -= row["bytes"]
    retired_names = {row["audio_filename"] for row in retired}
    coverage = Counter()
    for row in existing:
        if row["audio_filename"] not in retired_names:
            coverage.update(row["pitch_counts"])
    selected, remaining = [], budget
    candidates = list(candidates)
    while candidates:
        eligible = [row for row in candidates if 0 < row["bytes"] <= remaining and row["middle_fraction"] >= 0.75]
        if not eligible:
            break
        def benefit(row):
            gain = sum(math.sqrt(coverage[pitch] + count + 1) - math.sqrt(coverage[pitch] + 1)
                       for pitch, count in row["pitch_counts"].items() if low_note <= pitch <= high_note)
            return gain / row["bytes"]
        best = max(eligible, key=lambda row: (benefit(row), row["audio_filename"]))
        selected.append(best)
        candidates.remove(best)
        remaining -= best["bytes"]
        coverage.update(best["pitch_counts"])
    return retired, selected, coverage


def _download_verified(remote, file, stage):
    output = _safe_path(stage, file["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    for attempt in range(1, 4):
        try:
            crc, size = 0, 0
            with remote.open(file["member"]) as source, temporary.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            if size != file["bytes"] or f"{crc & 0xffffffff:08x}" != file["crc32"]:
                raise OSError("Downloaded training file differs from its ZIP size or CRC32.")
            temporary.replace(output)
            return output
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f"Retrying {file['path']}, attempt {attempt + 1}/3", flush=True)
            time.sleep(2 * attempt)


def refresh_middle_training(root, gigabytes=1.0, dry_run=False):
    if not math.isfinite(gigabytes) or gigabytes <= 0:
        raise ValueError("Middle replacement GB must be positive and finite.")
    root = Path(root).resolve()
    budget = round(gigabytes * 1_000_000_000)
    if not (root / CSV_NAME).is_file():
        raise ValueError("Download an initial MAESTRO training set before replacing recordings.")
    if shutil.disk_usage(root).free < budget + 100_000_000:
        raise OSError("Insufficient space for new data while retaining the inactive backups.")
    excluded_before = read_training_exclusions(root)
    with (root / CSV_NAME).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    archive_path = _midi_archive(root)
    print(f"Ranking official training recordings by focus coverage (MIDI {FOCUS_LOW_NOTE}-{FOCUS_HIGH_NOTE})...", flush=True)
    with zipfile.ZipFile(archive_path) as midi, remote_zip_class()(
            f"{BASE}/maestro-v3.0.0.zip", timeout=90, initial_buffer_size=512 * 1024) as audio:
        audio_names, midi_names = audio.namelist(), midi.namelist()
        existing, candidates = [], []
        for row in rows:
            if row["audio_filename"] in excluded_before:
                continue
            files = []
            for field, archive, names, kind in (("audio_filename", audio, audio_names, "audio"),
                                                ("midi_filename", midi, midi_names, "midi")):
                member = _member_name(names, row[field])
                info = archive.getinfo(member)
                local = _safe_path(root, row[field])
                files.append({"path": row[field], "member": member, "kind": kind,
                              "bytes": info.file_size, "compressed_bytes": info.compress_size,
                              "crc32": f"{info.CRC:08x}", "exists": local.is_file(),
                              "complete": local.is_file() and local.stat().st_size == info.file_size})
            complete = all(file["complete"] for file in files)
            size = sum(file["bytes"] for file in files)
            if not complete and (any(file["exists"] for file in files) or size > min(100_000_000, budget)):
                continue  # replacements must be genuinely new pairs, not overwrite partial/local data
            notes, _ = read_performance_bytes(midi.read(files[1]["member"]))
            counts = Counter(note.pitch for note in notes if 21 <= note.pitch <= 108)
            middle = sum(counts[pitch] for pitch in range(FOCUS_LOW_NOTE, FOCUS_HIGH_NOTE + 1))
            record = {**row, "files": files, "bytes": size, "pitch_counts": dict(counts),
                      "middle_count": middle, "middle_fraction": middle / sum(counts.values()) if counts else 0.0}
            (existing if complete else candidates).append(record)
        if len(existing) < 2:
            raise ValueError("At least two complete active training pairs are needed for replacement.")
        retired, selected, coverage = select_middle_refresh(existing, candidates, budget)
        if not selected or not retired:
            raise ValueError("The budget could not fit both old training pairs and new middle-rich pairs.")
        before_counts = Counter()
        for row in existing:
            before_counts.update(row["pitch_counts"])
        planned = sum(row["bytes"] for row in selected)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        report = {"source": "https://magenta.withgoogle.com/datasets/maestro", "version": "3.0.0", "split": "train",
                  "run_id": run_id, "budget_bytes": budget, "planned_new_bytes": planned,
                  "retired_bytes": sum(row["bytes"] for row in retired),
                  "estimated_transfer_bytes": sum(file["compressed_bytes"] for row in selected for file in row["files"]),
                  "middle_range": [FOCUS_LOW_NOTE, FOCUS_HIGH_NOTE], "minimum_new_middle_fraction": 0.75, "midi_index_sha256": MIDI_SHA256,
                  "active_training_before": len(existing), "active_training_after": len(existing) - len(retired) + len(selected),
                  "middle_events_before": sum(before_counts[p] for p in range(FOCUS_LOW_NOTE, FOCUS_HIGH_NOTE + 1)),
                  "middle_events_after": sum(coverage[p] for p in range(FOCUS_LOW_NOTE, FOCUS_HIGH_NOTE + 1)),
                  "previous_exclusions": sorted(excluded_before), "retired_records": retired, "new_records": selected,
                  "backup_policy": "Retired WAV/MIDI files stay in place, excluded only from newly opened training datasets.",
                  "status": "planned"}
        plan_path = root / "middle-refresh-plan.json"
        run_path = root / ".middle-refresh-manifests" / f"{run_id}.json"
        _write_json(plan_path, report)
        _write_json(run_path, report)
        print(f"Plan: add {len(selected)} new train recordings ({planned / 1e9:.3f} GB); "
              f"retire {len(retired)} ({report['retired_bytes'] / 1e9:.3f} GB); "
              f"middle strikes {report['middle_events_before']} -> {report['middle_events_after']}", flush=True)
        if dry_run:
            return report
        stage = _safe_path(root, f".middle-staging/{run_id}")
        for index, row in enumerate(selected, 1):
            for file in row["files"]:
                print(f"[{index}/{len(selected)}] {file['path']} ({file['bytes'] / 1e6:.1f} MB)", flush=True)
                _download_verified(audio if file["kind"] == "audio" else midi, file, stage)
        # Install only fully verified pairs, then atomically change the active training list.
        moved = []
        try:
            for row in selected:
                for file in row["files"]:
                    source, target = _safe_path(stage, file["path"]), _safe_path(root, file["path"])
                    if target.exists():
                        raise FileExistsError(f"Training destination appeared during download: {target}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(target)
                    moved.append((source, target))
            if read_training_exclusions(root) != excluded_before:
                raise RuntimeError("Training exclusions changed during download; active data was not replaced.")
            _write_json(root / EXCLUSIONS_NAME,
                        {"excluded_audio_filenames": sorted(excluded_before | {row["audio_filename"] for row in retired}),
                         "latest_refresh": run_id})
        except Exception:
            for source, target in reversed(moved):
                target.replace(source)
            raise
        report.update(status="complete", downloaded_new_bytes=planned)
        _write_json(plan_path, report)
        _write_json(run_path, report)
        print(f"Ready: {report['active_training_after']} active training recordings; retired files retained as backups.", flush=True)
        return report
