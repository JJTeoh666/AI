"""Restore the pinned, licensed desktop Basic Pitch model (~225 KiB)."""
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piano_ml.basic_pitch import MODEL_PATH, MODEL_SHA256, SOURCE_COMMIT


def main():
    base = f"https://raw.githubusercontent.com/spotify/basic-pitch/{SOURCE_COMMIT}/"
    with urlopen(base + "basic_pitch/saved_models/icassp_2022/nmp.onnx", timeout=60) as response:
        model = response.read()
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise ValueError("Downloaded model checksum mismatch.")
    with urlopen(base + "LICENSE", timeout=60) as response:
        license_text = response.read()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (MODEL_PATH.parent / "LICENSE").write_bytes(license_text)
    temporary = MODEL_PATH.with_suffix(".partial")
    temporary.write_bytes(model)
    temporary.replace(MODEL_PATH)
    print(f"Installed verified Basic Pitch model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
