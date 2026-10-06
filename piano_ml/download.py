"""Fetch selected MAESTRO v3 pairs via HTTP range requests."""

import csv
import json
import math
import random
import shutil
import time
import urllib.request
import importlib.util
from collections import Counter
from pathlib import Path, PurePosixPath

BASE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0"
CSV_NAME = "maestro-v3.0.0.csv"


def remote_zip_class():
    """Use an installed dependency, including this workspace's local .deps copy."""
    try:
        from remotezip import RemoteZip
        return RemoteZip
    except ImportError as error:
        local = Path(__file__).resolve().parent.parent / ".deps" / "remotezip.py"
        if not local.is_file():
            raise RuntimeError("Install requirements.txt to use selective download") from error
        spec = importlib.util.spec_from_file_location("piano_remotezip", local)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RemoteZip


def _member_name(names: list[str], relative: str) -> str:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe metadata path: {relative}")
    matches = [name for name in names if name == relative or name.endswith("/" + relative)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one archive member for {relative}; found {len(matches)}")
    return matches[0]


def download_subset(root: str | Path, counts: dict[str, int]) -> None:
    RemoteZip = remote_zip_class()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / CSV_NAME
    if not csv_path.exists():
        urllib.request.urlretrieve(f"{BASE}/{CSV_NAME}", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    selected = []
    for split in ("train", "validation", "test"):
        rows = sorted((r for r in all_rows if r["split"] == split),
                      key=lambda r: float(r["duration"]))
        selected.extend(rows[:counts[split]])
    if not selected:
        return
    for extension, field in (("midi", "midi_filename"), ("", "audio_filename")):
        archive = "maestro-v3.0.0-midi.zip" if extension else "maestro-v3.0.0.zip"
        needed = [row[field] for row in selected if not (root / row[field]).exists()]
        if not needed:
            continue
        print(f"Opening {archive} for {len(needed)} selected files", flush=True)
        with RemoteZip(f"{BASE}/{archive}", timeout=90) as remote:
            names = remote.namelist()
            for relative in needed:
                member = _member_name(names, relative)
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                print(f"Downloading {relative}", flush=True)
                with remote.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                temporary.replace(output)
    print(f"Ready: {len(selected)} paired recordings in {root}")


def download_additional(root: str | Path, gigabytes: float = 1.0, seed: int = 42,
                        dry_run: bool = False) -> dict:
    """Add approximately the requested GB of usable pairs, using archive sizes.

    Allocate 80/10/10 percent of additional disk bytes to the official splits.
    Sample different performances with a seeded shuffle and limit piece size
    so a small budget still includes a useful number of recordings.
    """
    RemoteZip = remote_zip_class()

    if not math.isfinite(gigabytes) or gigabytes <= 0:
        raise ValueError("Additional GB must be a positive finite number.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / CSV_NAME
    if not csv_path.exists():
        urllib.request.urlretrieve(f"{BASE}/{CSV_NAME}", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    budget = round(gigabytes * 1_000_000_000)
    if shutil.disk_usage(root).free < budget + 150_000_000:
        raise OSError("Insufficient free disk space for the requested data.")
    wav_url = f"{BASE}/maestro-v3.0.0.zip"
    midi_url = f"{BASE}/maestro-v3.0.0-midi.zip"
    print("Reading sizes from the official audio and MIDI archive indexes…", flush=True)
    with RemoteZip(wav_url, timeout=90, initial_buffer_size=512 * 1024) as wav_zip, \
            RemoteZip(midi_url, timeout=90, initial_buffer_size=512 * 1024) as midi_zip:
        wav_names, midi_names = wav_zip.namelist(), midi_zip.namelist()
        randomizer = random.Random(seed)
        selected = []
        for split, fraction in (("train", 0.8), ("validation", 0.1), ("test", 0.1)):
            remaining = round(budget * fraction)
            max_piece_bytes = min(100_000_000, remaining // (10 if split == "train" else 2))
            candidates = [row for row in rows if row["split"] == split]
            randomizer.shuffle(candidates)
            for row in candidates:
                files = []
                for field, archive, names, kind in (
                    ("audio_filename", wav_zip, wav_names, "audio"),
                    ("midi_filename", midi_zip, midi_names, "midi"),
                ):
                    relative = row[field]
                    member = _member_name(names, relative)
                    info = archive.getinfo(member)
                    local = root / relative
                    if local.is_file() and local.stat().st_size == info.file_size:
                        continue
                    files.append({"kind": kind, "path": relative, "member": member,
                                  "bytes": info.file_size, "compressed_bytes": info.compress_size,
                                  "crc32": f"{info.CRC:08x}"})
                size = sum(item["bytes"] for item in files)
                if not files or size > max_piece_bytes or size > remaining:
                    continue
                selected.append({"split": split, "composer": row["canonical_composer"],
                                 "title": row["canonical_title"], "seconds": float(row["duration"]),
                                 "files": files})
                remaining -= size
        planned = sum(file["bytes"] for record in selected for file in record["files"])
        compressed = sum(file["compressed_bytes"] for record in selected for file in record["files"])
        counts = {split: sum(record["split"] == split for record in selected)
                  for split in ("train", "validation", "test")}
        manifest = {"source": "https://magenta.withgoogle.com/datasets/maestro",
                    "version": "3.0.0", "requested_additional_bytes": budget,
                    "planned_additional_bytes": planned, "estimated_transfer_bytes": compressed,
                    "seed": seed, "split_counts": counts, "records": selected,
                    "status": "planned"}
        plan_path = root / "additional-download-plan.json"
        plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Plan: {len(selected)} recordings {counts}; {planned / 1e9:.3f} GB additional files, "
              f"approximately {compressed / 1e9:.3f} GB transfer", flush=True)
        if dry_run:
            return manifest
        completed = 0
        for index, record in enumerate(selected, 1):
            for file in record["files"]:
                remote = wav_zip if file["kind"] == "audio" else midi_zip
                output = root / file["path"]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                print(f"[{index}/{len(selected)}] {record['split']} {file['path']} "
                      f"({file['bytes'] / 1e6:.1f} MB)", flush=True)
                for attempt in range(1, 4):
                    try:
                        with remote.open(file["member"]) as source, temporary.open("wb") as target:
                            shutil.copyfileobj(source, target, 1024 * 1024)
                        if temporary.stat().st_size != file["bytes"]:
                            raise OSError("Downloaded file size differs from archive index.")
                        # ZipExtFile verifies the archive CRC while reading to EOF.
                        temporary.replace(output)
                        break
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        if attempt == 3:
                            raise
                        print(f"Retrying file, attempt {attempt + 1}/3…", flush=True)
                        time.sleep(2 * attempt)
                completed += file["bytes"]
            print(f"Completed {completed / 1e6:.1f}/{planned / 1e6:.1f} MB", flush=True)
        manifest["status"] = "complete"
        manifest["downloaded_additional_bytes"] = completed
        plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Ready: added {completed / 1e9:.3f} GB in {len(selected)} paired recordings to {root}", flush=True)
        return manifest


def download_validation(root: str | Path, total: int = 16, gigabytes: float = 1.0,
                        seed: int = 42, dry_run: bool = False) -> dict:
    """Expand official validation pairs with composer/year variety and a byte cap."""
    RemoteZip = remote_zip_class()

    if total < 1 or not math.isfinite(gigabytes) or gigabytes <= 0:
        raise ValueError("Validation count and download budget must be positive.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / CSV_NAME
    if not csv_path.exists():
        urllib.request.urlretrieve(f"{BASE}/{CSV_NAME}", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    budget = round(gigabytes * 1_000_000_000)
    wav_url, midi_url = f"{BASE}/maestro-v3.0.0.zip", f"{BASE}/maestro-v3.0.0-midi.zip"
    print("Reading official archive indexes for additional validation recordings…", flush=True)
    with RemoteZip(wav_url, timeout=90, initial_buffer_size=512 * 1024) as wav_zip, \
            RemoteZip(midi_url, timeout=90, initial_buffer_size=512 * 1024) as midi_zip:
        candidates, existing = [], []
        wav_names, midi_names = wav_zip.namelist(), midi_zip.namelist()
        for row in rows:
            files = []
            for field, archive, names, kind in (
                    ("audio_filename", wav_zip, wav_names, "audio"),
                    ("midi_filename", midi_zip, midi_names, "midi")):
                relative = row[field]
                member = _member_name(names, relative)
                info = archive.getinfo(member)
                local = root / relative
                if local.is_file() and local.stat().st_size == info.file_size:
                    continue
                files.append({"kind": kind, "path": relative, "member": member,
                              "bytes": info.file_size, "compressed_bytes": info.compress_size,
                              "crc32": f"{info.CRC:08x}"})
            if not files:
                existing.append(row)
            else:
                candidates.append({"split": "validation", "composer": row.get("canonical_composer", "Unknown"),
                                   "title": row.get("canonical_title", ""), "year": row.get("year", ""),
                                   "seconds": float(row["duration"]), "files": files})
        random.Random(seed).shuffle(candidates)
        composers = Counter(row.get("canonical_composer", "Unknown") for row in existing)
        years = Counter(row.get("year", "") for row in existing)
        selected, remaining = [], budget
        needed = max(0, total - len(existing))
        while len(selected) < needed:
            slots_after = needed - len(selected) - 1
            eligible = []
            for record in candidates:
                size = sum(file["bytes"] for file in record["files"])
                cheapest_rest = sorted(sum(file["bytes"] for file in other["files"])
                                       for other in candidates if other is not record)[:slots_after]
                if (size <= 100_000_000 and len(cheapest_rest) == slots_after
                        and size + sum(cheapest_rest) <= remaining):
                    eligible.append(record)
            if not eligible:
                raise ValueError(f"Cannot reach {total} validation pairs within {gigabytes:g} GB. "
                                 "Increase --validation-budget-gb or reduce --validation-total.")
            record = min(eligible, key=lambda item: (composers[item["composer"]], years[item["year"]]))
            candidates.remove(record)
            selected.append(record)
            remaining -= sum(file["bytes"] for file in record["files"])
            composers[record["composer"]] += 1
            years[record["year"]] += 1
        planned = budget - remaining
        if shutil.disk_usage(root).free < planned + 150_000_000:
            raise OSError("Insufficient free disk space for additional validation data.")
        manifest = {"source": "https://magenta.withgoogle.com/datasets/maestro", "version": "3.0.0",
                    "split": "validation", "existing_recordings": len(existing), "target_recordings": total,
                    "new_recordings": len(selected), "seed": seed, "budget_bytes": budget,
                    "planned_additional_bytes": planned,
                    "estimated_transfer_bytes": sum(file["compressed_bytes"] for record in selected for file in record["files"]),
                    "records": selected, "status": "planned"}
        plan_path = root / "validation-download-plan.json"
        plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Plan: add {len(selected)} validation recordings, {planned / 1e9:.3f} GB; "
              f"{len(existing)} already available", flush=True)
        if dry_run:
            return manifest
        for index, record in enumerate(selected, 1):
            for file in record["files"]:
                remote = wav_zip if file["kind"] == "audio" else midi_zip
                output = root / file["path"]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".partial")
                print(f"[{index}/{len(selected)}] validation {file['path']} ({file['bytes'] / 1e6:.1f} MB)", flush=True)
                for attempt in range(1, 4):
                    try:
                        with remote.open(file["member"]) as source, temporary.open("wb") as target:
                            shutil.copyfileobj(source, target, 1024 * 1024)
                        if temporary.stat().st_size != file["bytes"]:
                            raise OSError("Downloaded size differs from the archive index.")
                        temporary.replace(output)  # ZipExtFile checks CRC at EOF.
                        break
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        if attempt == 3:
                            raise
                        time.sleep(2 * attempt)
        manifest.update(status="complete", downloaded_additional_bytes=planned,
                        available_validation_recordings=len(existing) + len(selected))
        plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Ready: {manifest['available_validation_recordings']} complete validation pairs", flush=True)
        return manifest
