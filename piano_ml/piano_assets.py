"""Download the sampled piano and the official portable FluidSynth runtime."""

import hashlib
import json
import platform
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT / "assets"
SOUNDFONT = ASSETS / "piano" / "SalamanderGrandPiano.sf2"
FLUIDSYNTH_DIR = ASSETS / "fluidsynth"
PIANO_SOURCE = "https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html"
PIANO_URL = "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz"
PIANO_ARCHIVE_SHA256 = "15edb061d7ba60d58332f72dba8f8ce40988048cc703f935e6320f37d650e213"
PIANO_SHA256 = "712d0e681efbe5203a8014e9b3e84168f1908c82f2f6fb13bd2c77d6d72c70b7"
FLUIDSYNTH_VERSION = "2.6.1"
RUNTIME_URL = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.1/fluidsynth-v2.6.1-win10-x64-cpp11.zip"
RUNTIME_SHA256 = "fab7a2e4b85675b66970f97a39bbc239729c5e0f237198b5922a6a73cbc8677c"


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _download(url: str, path: Path, digest: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_name(path.name + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "PianoNotes/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length", 0))
            downloaded, last_update = 0, 0.0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_update >= 5:
                    size = f" / {total / 1e6:.1f} MB" if total else " MB"
                    print(f"Downloading {path.name}: {downloaded / 1e6:.1f}{size}", flush=True)
                    last_update = time.monotonic()
        if total and downloaded != total:
            raise OSError(f"Incomplete download: {path.name}")
        if digest and file_hash(temporary) != digest:
            raise OSError(f"Download checksum mismatch: {path.name}")
        temporary.replace(path)
    if digest and file_hash(path) != digest:
        raise OSError(f"Checksum mismatch: {path}; remove this archive and download it again.")
    return path


def install_piano() -> None:
    """Install local playback assets; never change global system libraries."""
    downloads = ASSETS / "downloads"
    manifest = {"piano": {"name": "Salamander Grand Piano V3+20200602",
        "author": "Alexander Holm", "instrument": "Yamaha C5",
        "source": PIANO_SOURCE, "download": PIANO_URL,
        "license": "CC BY 3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "conversion": "FreePats SoundFont conversion by roberto@zenvoid.org; see included source documentation.",
        "archive_sha256": PIANO_ARCHIVE_SHA256}}
    if not SOUNDFONT.is_file():
        archive = _download(PIANO_URL, downloads / "SalamanderGrandPiano-SF2-V3+20200602.tar.xz", PIANO_ARCHIVE_SHA256)
        SOUNDFONT.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:xz") as bank:
            fonts = [entry for entry in bank.getmembers() if entry.isfile() and entry.name.lower().endswith(".sf2")]
            if len(fonts) != 1:
                raise OSError("Expected one piano SoundFont in the official archive.")
            temporary = SOUNDFONT.with_suffix(".sf2.partial")
            with bank.extractfile(fonts[0]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            with temporary.open("rb") as handle:
                header = handle.read(12)
            if header[:4] != b"RIFF" or header[8:] != b"sfbk":
                raise OSError("Downloaded piano is not a valid SoundFont.")
            temporary.replace(SOUNDFONT)
            docs = SOUNDFONT.parent / "source-docs"
            docs.mkdir(exist_ok=True)
            for entry in bank.getmembers():
                if entry.isfile() and entry not in fonts and entry.size < 1024 * 1024:
                    # Copy documents as basenames; archive paths are never extracted.
                    with bank.extractfile(entry) as source, (docs / Path(entry.name).name).open("wb") as output:
                        shutil.copyfileobj(source, output)
    if file_hash(SOUNDFONT) != PIANO_SHA256:
        raise OSError("Piano SoundFont checksum mismatch; remove the damaged file and run download-piano again.")
    manifest["piano"].update(file=str(SOUNDFONT.relative_to(PROJECT)),
        bytes=SOUNDFONT.stat().st_size, sha256=file_hash(SOUNDFONT))
    if platform.system() == "Windows":
        if platform.architecture()[0] != "64bit":
            raise OSError("The bundled FluidSynth runtime requires 64-bit Python.")
        runtime = _download(RUNTIME_URL, downloads / "fluidsynth-v2.6.1-win10-x64-cpp11.zip", RUNTIME_SHA256)
        FLUIDSYNTH_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(runtime) as archive:
            for entry in archive.infolist():
                if not entry.is_dir() and "/bin/" in entry.filename:
                    data = archive.read(entry)  # ZipFile verifies the member CRC.
                    target = FLUIDSYNTH_DIR / Path(entry.filename).name
                    if not target.is_file() or target.read_bytes() != data:
                        temporary = target.with_name(target.name + ".partial")
                        temporary.write_bytes(data)
                        temporary.replace(target)
        license_url = "https://raw.githubusercontent.com/FluidSynth/fluidsynth/v2.6.1/LICENSE"
        _download(license_url, FLUIDSYNTH_DIR / "LICENSE.txt")
        manifest["runtime"] = {"name": "FluidSynth", "version": FLUIDSYNTH_VERSION,
            "source": "https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.1",
            "download": RUNTIME_URL, "sha256": RUNTIME_SHA256, "license": "LGPL-2.1-or-later"}
    else:
        print("Install libfluidsynth using your system package manager for sampled playback.", flush=True)
    (ASSETS / "piano-playback.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (SOUNDFONT.parent / "ATTRIBUTION.txt").write_text(
        "Salamander Grand Piano by Alexander Holm (Yamaha C5 samples).\n"
        "FreePats SoundFont conversion V3+20200602 by roberto@zenvoid.org.\n"
        f"Source: {PIANO_SOURCE}\n"
        "Creative Commons Attribution 3.0: https://creativecommons.org/licenses/by/3.0/\n"
        "Samples are rendered with FluidSynth; output level and room effects are set by this app.\n",
        encoding="utf-8")
    print(f"Sampled piano installed: {SOUNDFONT}", flush=True)


if __name__ == "__main__":
    install_piano()
