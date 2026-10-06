"""Select additional official training recordings by rare bass-note coverage."""

import csv
import hashlib
import json
import math
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from .download import BASE, CSV_NAME, _member_name, remote_zip_class
from .midi import read_performance_bytes

MIDI_SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _midi_archive(root):
    cache = root / ".bass-index"
    cache.mkdir(exist_ok=True)
    path = cache / "maestro-v3.0.0-midi.zip"
    digest = _sha256(path) if path.is_file() else None
    if digest != MIDI_SHA256:
        temporary = path.with_name(path.name + ".partial")
        print("Downloading the 56 MB official MIDI index to rank bass coverage…", flush=True)
        with urllib.request.urlopen(f"{BASE}/{path.name}", timeout=90) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        digest = _sha256(temporary)
        if digest != MIDI_SHA256:
            temporary.unlink(missing_ok=True)
            raise OSError("Official MIDI archive SHA256 mismatch.")
        temporary.replace(path)
    return path


def select_bass_recordings(candidates, existing_counts, budget):
    """Greedily cover scarce keys per added byte; reduce weight as coverage grows."""
    coverage = Counter(existing_counts)
    selected, remaining = [], budget
    candidates = list(candidates)
    while candidates:
        eligible = [row for row in candidates if 0 < row["bytes"] <= remaining and row["bass_count"] > 0]
        if not eligible:
            break
        def benefit(row):
            # Diminishing returns discourage one very repetitive bass pitch.
            gain = sum((math.sqrt(coverage[pitch] + count + 1) - math.sqrt(coverage[pitch] + 1))
                       * (2 if pitch <= 35 else 1) for pitch, count in row["pitch_counts"].items()
                       if 21 <= pitch <= 47)
            return gain / row["bytes"]
        best = max(eligible, key=lambda row: (benefit(row), row["audio_filename"]))
        selected.append(best)
        candidates.remove(best)
        remaining -= best["bytes"]
        coverage.update(best["pitch_counts"])
    return selected, coverage


def download_bass(root, gigabytes=1.0, dry_run=False):
    """Download bass-rich train pairs, keeping official validation/test untouched.

    Budget covers added usable pairs; the reusable 56 MB MIDI index is separate.
    Every selected file is checked against its ZIP size and CRC before commit.
    """
    if not math.isfinite(gigabytes) or gigabytes <= 0:
        raise ValueError("Bass download GB must be positive and finite.")
    RemoteZip = remote_zip_class()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    budget = round(gigabytes * 1_000_000_000)
    if shutil.disk_usage(root).free < budget + 100_000_000:
        raise OSError("Insufficient disk space for bass recordings and MIDI index.")
    csv_path = root / CSV_NAME
    if not csv_path.is_file():
        urllib.request.urlretrieve(f"{BASE}/{CSV_NAME}", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    archive_path = _midi_archive(root)
    print("Ranking official training performances by rare bass-note coverage…", flush=True)
    with zipfile.ZipFile(archive_path) as midi, RemoteZip(
            f"{BASE}/maestro-v3.0.0.zip", timeout=90, initial_buffer_size=512 * 1024) as audio:
        midi_names, audio_names = midi.namelist(), audio.namelist()
        candidates, existing = [], Counter()
        existing_recordings = 0
        for row in rows:
            midi_member = _member_name(midi_names, row["midi_filename"])
            audio_member = _member_name(audio_names, row["audio_filename"])
            notes, _ = read_performance_bytes(midi.read(midi_member))  # ZipFile verifies CRC.
            pitches = Counter(note.pitch for note in notes if 21 <= note.pitch <= 108)
            files = []
            for field, archive, member, kind in (("audio_filename", audio, audio_member, "audio"),
                                                ("midi_filename", midi, midi_member, "midi")):
                info = archive.getinfo(member)
                local = root / row[field]
                if local.is_file() and local.stat().st_size == info.file_size:
                    continue
                files.append({"path": row[field], "member": member, "kind": kind,
                              "bytes": info.file_size, "compressed_bytes": info.compress_size,
                              "crc32": f"{info.CRC:08x}"})
            if not files:
                existing.update(pitches)
                existing_recordings += 1
                continue
            candidates.append({**row, "files": files, "bytes": sum(f["bytes"] for f in files),
                               "pitch_counts": dict(pitches),
                               "bass_count": sum(pitches[p] for p in range(21, 48)),
                               "lowest_bass_count": sum(pitches[p] for p in range(21, 36))})
        selected, coverage = select_bass_recordings(candidates, existing, budget)
        planned = sum(row["bytes"] for row in selected)
        report = {"source": "https://magenta.withgoogle.com/datasets/maestro", "version": "3.0.0",
                  "split": "train", "budget_bytes": budget, "planned_additional_bytes": planned,
                  "midi_index_sha256": MIDI_SHA256, "existing_recordings": existing_recordings,
                  "added_recordings": len(selected), "status": "planned", "records": selected,
                  "bass_events_before": sum(existing[p] for p in range(21, 48)),
                  "bass_events_after": sum(coverage[p] for p in range(21, 48)),
                  "lowest_bass_events_before": sum(existing[p] for p in range(21, 36)),
                  "lowest_bass_events_after": sum(coverage[p] for p in range(21, 36))}
        plan_path = root / "bass-download-plan.json"
        plan_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Plan: {len(selected)} train recordings, {planned / 1e9:.3f} GB; "
              f"A0-B1 events {report['lowest_bass_events_before']} -> {report['lowest_bass_events_after']}", flush=True)
        if dry_run:
            return report
        for index, row in enumerate(selected, 1):
            for file in row["files"]:
                remote = audio if file["kind"] == "audio" else midi
                output = root / file["path"]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                print(f"[{index}/{len(selected)}] {file['path']} ({file['bytes'] / 1e6:.1f} MB)", flush=True)
                for attempt in range(1, 4):
                    try:
                        with remote.open(file["member"]) as source, temporary.open("wb") as target:
                            shutil.copyfileobj(source, target, 1024 * 1024)
                        if temporary.stat().st_size != file["bytes"]:
                            raise OSError("Downloaded file size differs from ZIP index.")
                        temporary.replace(output)
                        break
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        if attempt == 3:
                            raise
                        time.sleep(2 * attempt)
        report.update(status="complete", downloaded_additional_bytes=planned,
                      available_training_recordings=existing_recordings + len(selected))
        plan_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Ready: {report['available_training_recordings']} complete training recordings", flush=True)
        return report
