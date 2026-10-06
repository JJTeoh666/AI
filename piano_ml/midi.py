"""Small standard-MIDI reader for note intervals in seconds.

Note ends describe key releases. Sustain pedal intervals are returned separately.
"""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    velocity: int = 100


@dataclass(frozen=True)
class Pedal:
    start: float
    end: float


def _vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 128:
            return value, offset
    raise ValueError("Invalid MIDI variable-length quantity")


def read_performance(path: str | Path) -> tuple[list[Note], list[Pedal]]:
    return read_performance_bytes(Path(path).read_bytes())


def read_performance_bytes(data: bytes) -> tuple[list[Note], list[Pedal]]:
    """Parse an archive member without extracting candidate MIDI files."""
    if data[:4] != b"MThd":
        raise ValueError(f"Not a MIDI file: {path}")
    header_size = int.from_bytes(data[4:8], "big")
    n_tracks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division & 0x8000:
        raise ValueError("SMPTE MIDI time division is not supported")
    ticks_per_quarter = division
    offset = 8 + header_size
    events: list[tuple[int, int, int, int, int]] = []
    tempos: list[tuple[int, int]] = [(0, 500000)]
    last_tick = 0
    for _ in range(n_tracks):
        if data[offset:offset + 4] != b"MTrk":
            raise ValueError("Malformed MIDI track")
        size = int.from_bytes(data[offset + 4:offset + 8], "big")
        track = data[offset + 8:offset + 8 + size]
        offset += 8 + size
        pos = tick = 0
        running = None
        while pos < len(track):
            delta, pos = _vlq(track, pos)
            tick += delta
            status = track[pos]
            if status & 0x80:
                pos += 1
                if status < 0xF0:
                    running = status
            else:
                if running is None:
                    raise ValueError("MIDI running status without a prior status")
                status = running
            if status == 0xFF:
                kind = track[pos]
                pos += 1
                length, pos = _vlq(track, pos)
                payload = track[pos:pos + length]
                pos += length
                if kind == 0x51 and length == 3:
                    tempos.append((tick, int.from_bytes(payload, "big")))
            elif status in (0xF0, 0xF7):
                length, pos = _vlq(track, pos)
                pos += length
            elif status < 0xF0:
                kind = status & 0xF0
                channel = status & 0x0F
                pitch = track[pos]
                pos += 1
                velocity = 0
                if kind not in (0xC0, 0xD0):
                    velocity = track[pos]
                    pos += 1
                if channel != 9 and (kind in (0x80, 0x90) or (kind == 0xB0 and pitch == 64)):
                    events.append((tick, kind, channel, pitch, velocity))
            else:
                raise ValueError(f"Unsupported MIDI status 0x{status:02x}")
        last_tick = max(last_tick, tick)
    tempos.sort(key=lambda item: item[0])
    ticks = [0]
    seconds = [0.0]
    rates = [500000]
    for tick, rate in tempos[1:]:
        if tick == ticks[-1]:
            rates[-1] = rate
        else:
            seconds.append(seconds[-1] + (tick - ticks[-1]) * rates[-1] / (ticks_per_quarter * 1e6))
            ticks.append(tick)
            rates.append(rate)

    def to_seconds(tick: int) -> float:
        i = bisect_right(ticks, tick) - 1
        return seconds[i] + (tick - ticks[i]) * rates[i] / (ticks_per_quarter * 1e6)

    active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    pedal_down: dict[int, int] = {}
    notes, pedals = [], []
    for tick, kind, channel, pitch, velocity in sorted(events, key=lambda x: x[0]):
        if kind == 0xB0:
            if velocity >= 64:
                pedal_down.setdefault(channel, tick)
            elif channel in pedal_down:
                start = pedal_down.pop(channel)
                if tick > start:
                    pedals.append(Pedal(to_seconds(start), to_seconds(tick)))
            continue
        key = (channel, pitch)
        if kind == 0x90 and velocity > 0:
            active[key].append((tick, velocity))
        elif active[key]:
            start, strength = active[key].pop(0)
            if tick > start:
                notes.append(Note(pitch, to_seconds(start), to_seconds(tick), strength))
    final_tick = last_tick
    for (_, pitch), held in active.items():
        for start, strength in held:
            if final_tick > start:
                notes.append(Note(pitch, to_seconds(start), to_seconds(final_tick), strength))
    for start in pedal_down.values():
        if final_tick > start:
            pedals.append(Pedal(to_seconds(start), to_seconds(final_tick)))
    return sorted(notes, key=lambda n: (n.start, n.pitch)), sorted(pedals, key=lambda p: p.start)


def read_notes(path: str | Path) -> list[Note]:
    return read_performance(path)[0]
