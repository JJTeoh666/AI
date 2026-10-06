"""Stream PCM WAV playback with live, application-local volume on Windows."""

import ctypes
import math
import os
import threading
import wave
from pathlib import Path

import numpy as np


def playback_available() -> bool:
    return os.name == "nt"


def scale_pcm16(data: bytes, volume: float, gain_db: float = 0.0) -> bytes:
    """Apply playback level and gain, saturating peaks before 16-bit conversion."""
    if volume == 100 and gain_db == 0:
        return data
    if volume == 0:
        return bytes(len(data))
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32)
    gain = (volume / 100) * 10 ** (gain_db / 20)
    return np.clip(np.rint(samples * gain), -32768, 32767).astype("<i2").tobytes()


class _WaveFormat(ctypes.Structure):
    _pack_ = 2
    _fields_ = [("format", ctypes.c_uint16), ("channels", ctypes.c_uint16),
                ("rate", ctypes.c_uint32), ("bytes_per_second", ctypes.c_uint32),
                ("block_align", ctypes.c_uint16), ("bits", ctypes.c_uint16),
                ("extra_size", ctypes.c_uint16)]


class _WaveHeader(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_uint32),
                ("recorded", ctypes.c_uint32), ("user", ctypes.c_size_t),
                ("flags", ctypes.c_uint32), ("loops", ctypes.c_uint32),
                ("next", ctypes.c_void_p), ("reserved", ctypes.c_size_t)]


class _WaveOutput:
    """Own the WinMM stream and keep every queued buffer alive until returned.

    ABI: https://learn.microsoft.com/en-us/windows/win32/api/mmeapi/ns-mmeapi-wavehdr
    """

    def __init__(self, rate: int, channels: int):
        if not playback_available():
            raise OSError("Audio playback is supported on Windows.")
        self.lib = ctypes.WinDLL("winmm")
        pointer, uint = ctypes.c_void_p, ctypes.c_uint32
        header = ctypes.POINTER(_WaveHeader)
        signatures = {
            "waveOutOpen": [ctypes.POINTER(pointer), uint, ctypes.POINTER(_WaveFormat),
                            ctypes.c_size_t, ctypes.c_size_t, uint],
            "waveOutPrepareHeader": [pointer, header, uint],
            "waveOutWrite": [pointer, header, uint],
            "waveOutUnprepareHeader": [pointer, header, uint],
            "waveOutReset": [pointer], "waveOutClose": [pointer],
            "waveOutGetErrorTextW": [uint, ctypes.c_wchar_p, uint],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = uint, arguments
        self.handle = pointer()
        self.buffers = []
        align = channels * 2
        format = _WaveFormat(1, channels, rate, rate * align, align, 16, 0)
        self.check(self.lib.waveOutOpen(ctypes.byref(self.handle), 0xFFFFFFFF,
                                       ctypes.byref(format), 0, 0, 0))

    def check(self, code: int) -> None:
        if code:
            text = ctypes.create_unicode_buffer(256)
            self.lib.waveOutGetErrorTextW(code, text, len(text))
            raise OSError(text.value or f"Audio device error {code}")

    def write(self, data: bytes) -> None:
        buffer = ctypes.create_string_buffer(data)
        header = _WaveHeader(data=ctypes.cast(buffer, ctypes.c_void_p), length=len(data))
        self.check(self.lib.waveOutPrepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header)))
        # Keep memory alive even when write fails; close resets/unprepares it.
        self.buffers.append((buffer, header))
        self.check(self.lib.waveOutWrite(self.handle, ctypes.byref(header), ctypes.sizeof(header)))

    def collect(self) -> int:
        pending = []
        for buffer, header in self.buffers:
            if header.flags & 1:  # WHDR_DONE: the driver returned this buffer.
                self.check(self.lib.waveOutUnprepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header)))
            else:
                pending.append((buffer, header))
        self.buffers = pending
        return len(pending)

    def close(self) -> None:
        if not self.handle:
            return
        # Reset returns queued headers before any of their memory is released.
        self.check(self.lib.waveOutReset(self.handle))
        for _, header in self.buffers:
            self.check(self.lib.waveOutUnprepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header)))
        self.buffers.clear()
        self.check(self.lib.waveOutClose(self.handle))
        self.handle = ctypes.c_void_p()


class WavPlayer:
    """Bounded-memory playback; volume changes reach the next ~100 ms of audio."""

    def __init__(self, on_error=None):
        self.volume = 50.0
        self.gain_db = 0.0
        self.on_error = on_error
        self._cancel = threading.Event()
        self._thread = None

    def set_volume(self, percent: float) -> None:
        percent = float(percent)
        if not math.isfinite(percent):
            raise ValueError("Volume must be a finite percentage.")
        self.volume = max(0.0, min(100.0, percent))

    def set_gain(self, decibels: float) -> None:
        decibels = float(decibels)
        if not math.isfinite(decibels):
            raise ValueError("Gain must be finite.")
        self.gain_db = max(0.0, min(24.0, decibels))

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, path: str | Path) -> float:
        self.stop()
        audio = wave.open(str(path), "rb")
        try:
            if (audio.getsampwidth() != 2 or audio.getcomptype() != "NONE"
                    or audio.getnchannels() not in (1, 2)):
                raise ValueError("Playback requires mono or stereo 16-bit PCM WAV.")
            duration = audio.getnframes() / audio.getframerate()
            output = _WaveOutput(audio.getframerate(), audio.getnchannels())
        except BaseException:
            audio.close()
            raise
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._stream, args=(audio, output, self._cancel), daemon=True)
        try:
            self._thread.start()
        except BaseException:
            output.close()
            audio.close()
            self._thread = None
            raise
        return duration

    def _stream(self, audio, output, cancel) -> None:
        error = None
        try:
            frames = max(1, round(audio.getframerate() * 0.05))
            eof = False
            while not cancel.is_set():
                pending = output.collect()
                while not eof and pending < 2 and not cancel.is_set():
                    data = audio.readframes(frames)
                    if not data:
                        eof = True
                    else:
                        output.write(scale_pcm16(data, self.volume, self.gain_db))
                        pending += 1
                if eof and not pending:
                    break
                cancel.wait(0.005)
        except Exception as problem:
            error = problem
        finally:
            try:
                output.close()
            except Exception as problem:
                error = error or problem
            audio.close()
        if error is not None and not cancel.is_set() and self.on_error is not None:
            self.on_error(str(error))

    def stop(self) -> None:
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise OSError("Audio device did not stop in time.")
            self._thread = None
