"""Snapshot held-out files, then verify a completed middle-focus data replacement."""

import argparse
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/maestro")
    parser.add_argument("--baseline", default="diagnostics/training-focus-before-2026-10-07.json")
    parser.add_argument("--output", default="diagnostics/training-focus-refresh-2026-10-07.json")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    root = Path(args.data).resolve()
    baseline = Path(args.baseline)
    if args.snapshot:
        with (root / "maestro-v3.0.0.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        available = [row for row in rows if all((root / row[field]).is_file()
                                               for field in ("audio_filename", "midi_filename"))]
        heldout = [root / row[field] for row in available if row["split"] != "train"
                   for field in ("audio_filename", "midi_filename")]
        protected = [root / "maestro-v3.0.0.csv", *heldout,
                     Path("checkpoints/piano-v8.pt").resolve(), Path("checkpoints/piano-v8.last.pt").resolve()]
        payload = {"protected_sha256": {str(path): digest(path) for path in protected if path.is_file()},
                   "split_counts": {split: sum(row["split"] == split for row in available)
                                    for split in ("train", "validation", "test")}}
        baseline.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"snapshot": str(baseline), "split_counts": payload["split_counts"]}))
        return
    from piano_ml.data import MaestroWindows, load_records
    from piano_ml.midi import read_performance
    from piano_ml.middle_data import read_training_exclusions
    before = json.loads(baseline.read_text(encoding="utf-8"))
    report = json.loads((root / "middle-refresh-plan.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete", "Refresh has not completed."
    changed = [name for name, expected in before["protected_sha256"].items() if digest(Path(name)) != expected]
    assert not changed, f"Protected files changed: {changed}"
    verified_files, checked_windows = 0, 0
    low, high = report["middle_range"]
    data = MaestroWindows(root, "train", seconds=4, windows_per_file=1, random_windows=True, multi_target=True)
    for row in report["new_records"]:
        for file in row["files"]:
            path, crc = root / file["path"], 0
            assert path.stat().st_size == file["bytes"]
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(block, crc)
            assert f"{crc & 0xffffffff:08x}" == file["crc32"]
            verified_files += 1
        notes, _ = read_performance(root / row["midi_filename"])
        notes = [note for note in notes if 21 <= note.pitch <= 108]
        assert sum(low <= note.pitch <= high for note in notes) / len(notes) >= report["minimum_new_middle_fraction"]
        index = next(i for i, candidate in enumerate(data.records) if candidate["audio_filename"] == row["audio_filename"])
        audio, labels = data[index]
        assert audio.shape == (64000,) and labels["frame"].shape == (201, 88)
        checked_windows += 1
    excluded = read_training_exclusions(root)
    for row in report["retired_records"]:
        assert row["audio_filename"] in excluded
        assert all((root / file["path"]).stat().st_size == file["bytes"] for file in row["files"])
    counts = {split: len(load_records(root, split)) for split in ("train", "validation", "test")}
    assert counts["train"] == report["active_training_after"]
    assert all(counts[split] == before["split_counts"][split] for split in ("validation", "test"))
    payload = {key: report[key] for key in ("source", "run_id", "middle_range", "downloaded_new_bytes", "retired_bytes",
                                           "middle_events_before", "middle_events_after", "active_training_before", "active_training_after")}
    payload.update(new_recordings=len(report["new_records"]), retired_recordings=len(report["retired_records"]),
                   split_counts=counts, verified_new_files=verified_files, checked_training_windows=checked_windows,
                   protected_sha256_unchanged=True, protected_files=len(before["protected_sha256"]),
                   original_files_retained=True, selection_metric="note_macro_f0123",
                   plan=str(root / "middle-refresh-plan.json"), exclusions=str(root / "training-exclusions.json"))
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
