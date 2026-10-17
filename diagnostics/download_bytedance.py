"""Download and verify the official 172 MB note-and-pedal checkpoint."""

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piano_ml.bytedance import MODEL_BYTES, MODEL_MD5, MODEL_SHA256, MODEL_PATH, MODEL_URL, verify_checkpoint


def main():
    if MODEL_PATH.is_file():
        try:
            verify_checkpoint()
            print(f"Verified: {MODEL_PATH}")
            return
        except ValueError:
            pass
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".pth.part")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "PianoNotesStudio/1.0"})
        checksum, sha256, downloaded, reported = hashlib.md5(), hashlib.sha256(), 0, -1
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                checksum.update(chunk)
                sha256.update(chunk)
                downloaded += len(chunk)
                percent = downloaded * 100 // MODEL_BYTES
                if percent // 5 != reported:
                    print(f"Downloading ByteDance: {percent}%", flush=True)
                    reported = percent // 5
        if downloaded != MODEL_BYTES or checksum.hexdigest() != MODEL_MD5 or sha256.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded checkpoint does not match the official Zenodo checksum.")
        partial.replace(MODEL_PATH)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Verified: {MODEL_PATH}")


if __name__ == "__main__":
    main()
