"""Offline stereo piano rendering through the FluidSynth C API."""

import ctypes
import ctypes.util
import math
import os
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .piano_assets import FLUIDSYNTH_DIR, SOUNDFONT

PIANO_RELEASE_SECONDS = 2.5


def _library_path() -> str | None:
    bundled = FLUIDSYNTH_DIR / "libfluidsynth-3.dll"
    if os.name == "nt" and bundled.is_file():
        return str(bundled)
    return ctypes.util.find_library("fluidsynth")


def sampled_piano_available() -> bool:
    return SOUNDFONT.is_file() and _library_path() is not None


class SampledPiano:
    """One synthesizer per render worker; no audio device or MIDI driver needed."""

    def __init__(self, sample_rate: int, soundfont: Path = SOUNDFONT):
        library = _library_path()
        if library is None or not soundfont.is_file():
            raise OSError("Sampled piano is missing. Run: python -m piano_ml download-piano")
        self.settings = self.synth = None
        self.dll_directory = None
        if os.name == "nt" and Path(library).is_file():
            self.dll_directory = os.add_dll_directory(str(Path(library).parent))
        try:
            self.lib = ctypes.CDLL(library)
            ptr, integer = ctypes.c_void_p, ctypes.c_int
            signatures = {
                "new_fluid_settings": (ptr, []),
                "delete_fluid_settings": (None, [ptr]),
                "fluid_settings_setnum": (integer, [ptr, ctypes.c_char_p, ctypes.c_double]),
                "fluid_settings_setint": (integer, [ptr, ctypes.c_char_p, integer]),
                "new_fluid_synth": (ptr, [ptr]),
                "delete_fluid_synth": (None, [ptr]),
                "fluid_synth_sfload": (integer, [ptr, ctypes.c_char_p, integer]),
                "fluid_synth_program_select": (integer, [ptr, integer, integer, integer, integer]),
                "fluid_synth_noteon": (integer, [ptr, integer, integer, integer]),
                "fluid_synth_noteoff": (integer, [ptr, integer, integer]),
                "fluid_synth_cc": (integer, [ptr, integer, integer, integer]),
                "fluid_synth_write_float": (integer, [ptr, integer, ptr, integer, integer, ptr, integer, integer]),
            }
            for name, (result, arguments) in signatures.items():
                function = getattr(self.lib, name)
                function.restype, function.argtypes = result, arguments
            self.settings = self.lib.new_fluid_settings()
            if not self.settings:
                raise OSError("Cannot allocate piano synthesizer settings.")
            for name, value in (("synth.sample-rate", sample_rate), ("synth.gain", 0.6),
                                ("synth.reverb.room-size", 0.35), ("synth.reverb.damp", 0.4),
                                ("synth.reverb.width", 0.8), ("synth.reverb.level", 0.12)):
                if self.lib.fluid_settings_setnum(self.settings, name.encode(), value) != 0:
                    raise OSError(f"Cannot set piano option: {name}")
            for name, value in (("synth.polyphony", 256), ("synth.chorus.active", 0),
                                ("synth.reverb.active", 1), ("synth.cpu-cores", 1),
                                ("synth.lock-memory", 0)):
                if self.lib.fluid_settings_setint(self.settings, name.encode(), value) != 0:
                    raise OSError(f"Cannot set piano option: {name}")
            # The bundled 2.6 runtime supports a stereo limiter. Keep older
            # system libraries usable, with a conservative gain when unavailable.
            if self.lib.fluid_settings_setint(self.settings, b"synth.limiter.active", 1) == 0:
                self.lib.fluid_settings_setnum(self.settings, b"synth.limiter.output-limit", 0.96)
                self.lib.fluid_settings_setnum(self.settings, b"synth.limiter.attack", 1.0)
                self.lib.fluid_settings_setnum(self.settings, b"synth.limiter.link-channels", 1.0)
            else:
                self.lib.fluid_settings_setnum(self.settings, b"synth.gain", 0.25)
            self.synth = self.lib.new_fluid_synth(self.settings)
            if not self.synth:
                raise OSError("Cannot initialize the piano synthesizer.")
            bank = self.lib.fluid_synth_sfload(self.synth, str(soundfont).encode("utf-8"), 0)
            if bank < 0 or self.lib.fluid_synth_program_select(self.synth, 0, bank, 0, 0) != 0:
                raise OSError("Cannot load the Salamander piano instrument.")
        except BaseException:
            self.close()
            raise

    def event(self, kind: str, key: int, value: int) -> None:
        if kind == "on":
            status = self.lib.fluid_synth_noteon(self.synth, 0, key, value)
        elif kind == "off":
            status = self.lib.fluid_synth_noteoff(self.synth, 0, key)
        else:
            status = self.lib.fluid_synth_cc(self.synth, 0, key, value)
        if status != 0:
            raise OSError(f"Piano event failed: {kind} {key}")

    def render(self, frames: int) -> np.ndarray:
        audio = np.empty((frames, 2), dtype=np.float32)
        status = self.lib.fluid_synth_write_float(self.synth, frames,
            audio.ctypes.data, 0, 2, audio.ctypes.data, 1, 2)
        if status != 0:
            raise OSError("Piano audio rendering failed.")
        return audio

    def close(self) -> None:
        if self.synth:
            self.lib.delete_fluid_synth(self.synth)
            self.synth = None
        if self.settings:
            self.lib.delete_fluid_settings(self.settings)
            self.settings = None
        if self.dll_directory is not None:
            self.dll_directory.close()
            self.dll_directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _events(result: dict, sample_rate: int) -> list[tuple]:
    """Sample-timed MIDI events; release old strikes before repeating a key."""
    events = []
    duration = float(result["duration"])
    next_start = {}
    for note in sorted(result["notes"], key=lambda value: value["start"], reverse=True):
        pitch = int(note["pitch"])
        end = min(note["end"], next_start.get(pitch, duration))
        next_start[pitch] = note["start"]
        # Keep sub-sample notes valid, and avoid note-offs preceding their strike.
        start_frame = round(note["start"] * sample_rate)
        end_frame = max(start_frame + 1, round(end * sample_rate))
        events.append((start_frame, 3, "on", pitch, int(note.get("velocity", 100))))
        events.append((end_frame, 2, "off", pitch, 0))
    # Union overlapping sustain intervals before sending CC64 events.
    pedals = []
    for interval in sorted(result.get("pedals", []), key=lambda value: value["start"]):
        if pedals and interval["start"] <= pedals[-1][1]:
            pedals[-1][1] = max(pedals[-1][1], interval["end"])
        else:
            pedals.append([interval["start"], interval["end"]])
    for start, end in pedals:
        events.append((round(start * sample_rate), 1, "cc", 64, 127))
        events.append((round(end * sample_rate), 0, "cc", 64, 0))
    events.append((round(duration * sample_rate), 4, "cc", 64, 0))
    return sorted(events)


def render_sampled_wav(result: dict, output: str | Path, sample_rate: int = 44100,
                       progress: Callable[[int, int], None] | None = None,
                       cancel: threading.Event | None = None) -> Path:
    """Stream stereo PCM, preserving note events, velocities and sustain pedal."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    total_frames = round((float(result["duration"]) + (PIANO_RELEASE_SECONDS if result["notes"] else 0)) * sample_rate)
    total_blocks = math.ceil(total_frames / sample_rate)
    events = _events(result, sample_rate)
    next_event, position = 0, 0
    try:
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Result audio rendering cancelled.")
        with SampledPiano(sample_rate) as piano, wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            while position < total_frames:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("Result audio rendering cancelled.")
                while next_event < len(events) and events[next_event][0] <= position:
                    _, _, kind, key, value = events[next_event]
                    piano.event(kind, key, value)
                    next_event += 1
                # Short render calls keep cancellation responsive even for long recordings.
                end = min(position + 4096, total_frames)
                if next_event < len(events):
                    end = min(end, events[next_event][0])
                audio = piano.render(end - position)
                pcm = (np.clip(audio, -0.999, 0.999) * 32767).astype("<i2")
                wav.writeframes(pcm.tobytes())
                if progress is not None and (end // sample_rate != position // sample_rate or end == total_frames):
                    progress(min(total_blocks, math.ceil(end / sample_rate)), total_blocks)
                position = end
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Result audio rendering cancelled.")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output
