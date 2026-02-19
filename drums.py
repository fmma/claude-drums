#!/usr/bin/env python3
"""Generate and play drum patterns from text notation."""

import os
import sys
import re
import tty
import time
import wave
import select
import struct
import termios
import argparse
import numpy as np
import sounddevice as sd

SR = 44100

# Instrument abbreviation -> MIDI note number (used as sample key)
GM_DRUMS = {
    "BD": 36, "KD": 36,
    "RS": 37,
    "SD": 38,
    "CL": 39,
    "LT": 41, "FT": 41, "T3": 41,
    "HH": 42, "CH": 42,
    "PH": 44,
    "MT": 45, "T2": 45,
    "OH": 46,
    "HT": 48, "T1": 48,
    "CR": 49,
    "RD": 51,
    "RB": 53,
    "TM": 54,
    "SP": 55,
    "CB": 56,
    "C2": 57,
}


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _t(dur):
    return np.arange(int(SR * dur)) / SR


def _hp(signal, cutoff=200):
    """Simple first-order high-pass filter."""
    rc = 1.0 / (2.0 * np.pi * cutoff)
    dt = 1.0 / SR
    alpha = rc / (rc + dt)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * (out[i - 1] + signal[i] - signal[i - 1])
    return out


def _bandpass(signal, lo, hi):
    """Crude bandpass via FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec = np.fft.rfft(signal)
    mask = np.zeros_like(freqs)
    band = (freqs >= lo) & (freqs <= hi)
    mask[band] = 1.0
    # Smooth edges to avoid clicks
    edge = max(1, int(n * 30 / SR))
    for i in range(len(mask)):
        if mask[i] == 1.0:
            for j in range(1, edge + 1):
                if i - j >= 0 and mask[i - j] == 0:
                    mask[i - j] = max(mask[i - j], 1.0 - j / edge)
                if i + j < len(mask) and mask[i + j] == 0:
                    mask[i + j] = max(mask[i + j], 1.0 - j / edge)
    return np.fft.irfft(spec * mask, n)


def _kick():
    t = _t(0.55)
    # Sub body: pitch sweep from ~150 Hz down to 42 Hz
    freq = 110 * np.exp(-t * 10) + 42
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t * 5.5)
    # Second harmonic adds warmth
    body += np.sin(phase * 2) * np.exp(-t * 9) * 0.25
    # Transient click: short burst of high-freq
    click_freq = 4000 * np.exp(-t * 120) + 200
    click = np.sin(2 * np.pi * np.cumsum(click_freq) / SR) * np.exp(-t * 80) * 0.5
    # Noise thump at the very start
    thump = np.random.randn(len(t)) * np.exp(-t * 60) * 0.15
    out = body + click + thump
    # Slight compression feel: soft-clip the transient
    out = np.tanh(out * 1.3) * 0.85
    return out


def _snare():
    t = _t(0.35)
    # Drum body: two resonant modes
    freq1 = 200 * np.exp(-t * 6) + 150
    body1 = np.sin(2 * np.pi * np.cumsum(freq1) / SR) * np.exp(-t * 12) * 0.45
    body2 = np.sin(2 * np.pi * 340 * t) * np.exp(-t * 18) * 0.2
    # Snare wires: band-limited noise (2-8 kHz)
    noise = np.random.randn(len(t))
    wires = _bandpass(noise, 2000, 9000) * np.exp(-t * 10) * 1.2
    # Attack transient
    transient = np.random.randn(len(t)) * np.exp(-t * 55) * 0.6
    out = body1 + body2 + wires + transient
    return np.tanh(out * 1.1) * 0.8


def _rimshot():
    t = _t(0.12)
    # Sharp stick impact: high-frequency ring
    tone1 = np.sin(2 * np.pi * 1720 * t) * np.exp(-t * 35) * 0.5
    tone2 = np.sin(2 * np.pi * 940 * t) * np.exp(-t * 30) * 0.35
    # Short noise burst
    noise = np.random.randn(len(t)) * np.exp(-t * 60) * 0.4
    return tone1 + tone2 + noise


def _clap():
    t = _t(0.35)
    noise = np.random.randn(len(t))
    # Multiple micro-hits spread over ~30ms (people clapping slightly out of sync)
    env = np.zeros(len(t))
    for off in [0.0, 0.008, 0.018, 0.025]:
        i = int(off * SR)
        n = int(0.006 * SR)
        if i + n < len(t):
            env[i:i + n] += 0.8
    # Tail reverb-like decay
    env += np.exp(-t * 12) * 0.5
    out = noise * env
    # Bandpass to make it sound more natural (cut lows and extreme highs)
    out = _bandpass(out, 500, 7000)
    return out * 0.65


def _hihat_closed():
    t = _t(0.1)
    # Inharmonic metallic partials (hi-hats have non-integer frequency ratios)
    freqs = [687, 1327, 1953, 2687, 3573, 4340, 5213, 6427]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                for f in freqs) / len(freqs)
    noise = np.random.randn(len(t)) * 0.4
    env = np.exp(-t * 45) * 0.7 + np.exp(-t * 120) * 0.3
    return (metal * 0.5 + noise) * env * 0.4


def _hihat_open():
    t = _t(0.6)
    freqs = [687, 1327, 1953, 2687, 3573, 4340, 5213, 6427]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                for f in freqs) / len(freqs)
    noise = np.random.randn(len(t)) * 0.35
    env = np.exp(-t * 5.5)
    return (metal * 0.55 + noise) * env * 0.4


def _pedal_hihat():
    t = _t(0.07)
    freqs = [687, 1327, 1953, 2687, 3573]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                for f in freqs) / len(freqs)
    noise = np.random.randn(len(t)) * 0.35
    env = np.exp(-t * 55)
    return (metal * 0.4 + noise) * env * 0.28


def _tom(freq_start, freq_end, dur=0.5):
    t = _t(dur)
    # Pitch sweep body
    freq = (freq_start - freq_end) * np.exp(-t * 6) + freq_end
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t * 5.5)
    # Second harmonic
    body += np.sin(phase * 2) * np.exp(-t * 8) * 0.15
    # Stick attack
    attack = np.random.randn(len(t)) * np.exp(-t * 40) * 0.25
    out = body + attack
    return np.tanh(out * 1.2) * 0.7


def _crash():
    t = _t(2.0)
    # Dense inharmonic partials for complex cymbal shimmer
    freqs = [423, 637, 892, 1210, 1583, 2017, 2534, 3173, 3842, 4657, 5520]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                * np.exp(-t * (1.5 + i * 0.15))
                for i, f in enumerate(freqs)) / len(freqs)
    noise = np.random.randn(len(t))
    noise_env = np.exp(-t * 3) * 0.5 + np.exp(-t * 15) * 0.3
    # Fast attack, slow decay
    attack = np.exp(-t * 50) * 0.4
    out = (metal * 0.5 + noise * noise_env) + attack * noise
    return out * 0.35


def _ride():
    t = _t(1.5)
    # Defined stick "ping" + wash
    ping = np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 18) * 0.25
    freqs = [620, 1190, 1840, 2730, 3650, 4480]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                * np.exp(-t * (2.0 + i * 0.2))
                for i, f in enumerate(freqs)) / len(freqs)
    noise = np.random.randn(len(t)) * np.exp(-t * 4) * 0.15
    return ping + metal * 0.4 + noise


def _ride_bell():
    t = _t(1.2)
    # Prominent bell tone: strong fundamentals
    tone1 = np.sin(2 * np.pi * 845 * t) * 0.5
    tone2 = np.sin(2 * np.pi * 1690 * t) * 0.3
    tone3 = np.sin(2 * np.pi * 2535 * t) * 0.15
    env = np.exp(-t * 3)
    return (tone1 + tone2 + tone3) * env * 0.35


def _tambourine():
    t = _t(0.3)
    # Jingles: many high inharmonic partials
    freqs = [3100, 4250, 5400, 6800, 8200, 9600]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                for f in freqs) / len(freqs)
    noise = np.random.randn(len(t)) * 0.3
    # Initial slap + jingle decay
    env = np.exp(-t * 25) * 0.4 + np.exp(-t * 8) * 0.6
    return (metal * 0.5 + noise) * env * 0.3


def _splash():
    t = _t(0.7)
    freqs = [580, 1150, 1820, 2470, 3280, 4120, 5100]
    metal = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
                * np.exp(-t * (4 + i * 0.3))
                for i, f in enumerate(freqs)) / len(freqs)
    noise = np.random.randn(len(t)) * np.exp(-t * 6) * 0.4
    attack = np.random.randn(len(t)) * np.exp(-t * 50) * 0.3
    return (metal * 0.5 + noise + attack) * 0.38


def _cowbell():
    t = _t(0.35)
    # Two resonant modes, slight detuning for realism
    tone1 = np.sin(2 * np.pi * 587 * t) * 0.6
    tone2 = np.sin(2 * np.pi * 845 * t) * 0.4
    # Add subtle odd harmonics for metallic edge
    tone3 = np.sin(2 * np.pi * 1690 * t) * 0.1
    env = np.exp(-t * 10) * 0.6 + np.exp(-t * 25) * 0.4
    out = (tone1 + tone2 + tone3) * env
    return np.tanh(out * 2) * 0.22


def make_samples():
    return {
        36: _kick(),
        37: _rimshot(),
        38: _snare(),
        39: _clap(),
        41: _tom(110, 65, dur=0.6),
        42: _hihat_closed(),
        44: _pedal_hihat(),
        45: _tom(160, 95, dur=0.5),
        46: _hihat_open(),
        48: _tom(220, 130, dur=0.45),
        49: _crash(),
        51: _ride(),
        53: _ride_bell(),
        54: _tambourine(),
        55: _splash(),
        56: _cowbell(),
        57: _crash(),
    }


def load_kit(dir_path):
    """Load WAV drum samples from a directory.

    Expects files named by instrument abbreviation (e.g. BD.wav, SD.wav).
    Returns a {midi_note: numpy_array} dict, same format as make_samples().
    Falls back to synthesized sounds for missing instruments.
    """
    samples = make_samples()

    # Build reverse map: abbreviation -> midi note
    abbrev_to_note = {}
    for abbr, note in GM_DRUMS.items():
        abbrev_to_note[abbr] = note

    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".wav"):
            continue
        abbr = os.path.splitext(fname)[0].upper()
        if abbr not in abbrev_to_note:
            continue

        path = os.path.join(dir_path, fname)
        try:
            with wave.open(path, "r") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            # Decode to float64
            if sampwidth == 2:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
                data /= 32768.0
            elif sampwidth == 3:
                # 24-bit: pad each 3-byte sample to 4 bytes (vectorized)
                raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                padded = np.zeros((len(raw_bytes), 4), dtype=np.uint8)
                padded[:, 1:] = raw_bytes  # place in upper 3 bytes for sign extension
                data = padded.view(np.int32).flatten().astype(np.float64)
                data /= 2147483648.0
            elif sampwidth == 4:
                data = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
                data /= 2147483648.0
            else:
                print(f"  Skipping {fname}: unsupported bit depth ({sampwidth * 8}-bit)", file=sys.stderr)
                continue

            # Convert to mono by averaging channels
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)

            # Resample to SR if needed
            if framerate != SR:
                duration = len(data) / framerate
                n_out = int(duration * SR)
                x_old = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, n_out)
                data = np.interp(x_new, x_old, data)

            samples[abbrev_to_note[abbr]] = data

        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}", file=sys.stderr)

    return samples


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

def get_velocity(char):
    if char in ("x", "o"):
        return 100
    if char in ("X", "O"):
        return 127
    if char in (".", "g"):
        return 50
    return 0


def parse_pattern(text):
    lines = text.strip().split("\n")
    bpm = 120
    swing = 0
    repeat = None
    title = ""
    tracks = []
    steps_per_bar = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            continue

        m = re.match(r"BPM:\s*(\d+)", stripped, re.IGNORECASE)
        if m:
            bpm = int(m.group(1))
            continue
        m = re.match(r"Swing:\s*(\d+)", stripped, re.IGNORECASE)
        if m:
            swing = int(m.group(1))
            continue
        m = re.match(r"Repeat:\s*(\d+)", stripped, re.IGNORECASE)
        if m:
            repeat = int(m.group(1))
            continue
        m = re.match(r"Title:\s*(.+)", stripped, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            continue

        # Old format: INST |pattern|
        m = re.match(r"([A-Za-z]\w{1,2})\s*\|(.+)", stripped)
        if m:
            instrument = m.group(1).upper()
            rest = m.group(2).rstrip("|")
            bars = rest.split("|")
            if bars:
                steps_per_bar = max(steps_per_bar or 0, len(bars[0]))
            tracks.append({
                "instrument": instrument,
                "steps": "".join(bars),
                "num_bars": len(bars),
            })
            continue

        # New format: pattern INST (label at end, no pipes)
        m = re.match(r"^([xoXO.g\-]+)\s+([A-Za-z]\w{1,2})\s*$", stripped)
        if m:
            pattern = m.group(1)
            instrument = m.group(2).upper()
            steps_per_bar = max(steps_per_bar or 0, len(pattern))
            tracks.append({
                "instrument": instrument,
                "steps": pattern,
                "num_bars": 1,
            })

    return {
        "title": title,
        "bpm": bpm,
        "swing": swing,
        "repeat": repeat,
        "tracks": tracks,
        "steps_per_bar": steps_per_bar or 16,
    }


def extract_patterns(text):
    blocks = re.findall(r"```drums\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return [parse_pattern(b) for b in blocks]
    return [parse_pattern(text)]


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def mix_patterns(patterns, samples):
    # Calculate total duration in seconds
    total_secs = 0.0
    for p in patterns:
        spb = p["steps_per_bar"]
        step_s = (60.0 / p["bpm"]) * (4.0 / spb)
        max_steps = max((len(t["steps"]) for t in p["tracks"]), default=0)
        reps = p["repeat"] or 1
        total_secs += max_steps * step_s * reps

    audio = np.zeros(int(SR * total_secs))
    time_offset = 0.0

    for p in patterns:
        spb = p["steps_per_bar"]
        step_s = (60.0 / p["bpm"]) * (4.0 / spb)
        swing_pct = p["swing"] / 100.0
        max_steps = max((len(t["steps"]) for t in p["tracks"]), default=0)
        pat_dur = max_steps * step_s
        reps = p["repeat"] or 1

        for _ in range(reps):
            for t in p["tracks"]:
                inst = t["instrument"]
                if inst not in GM_DRUMS:
                    print(f"Warning: unknown instrument '{inst}'", file=sys.stderr)
                    continue
                note = GM_DRUMS[inst]
                if note not in samples:
                    continue
                sample = samples[note]

                for i, char in enumerate(t["steps"]):
                    vel = get_velocity(char)
                    if vel > 0:
                        t_sec = time_offset + i * step_s
                        if swing_pct > 0 and i % 2 == 1:
                            t_sec += swing_pct * step_s
                        start = int(t_sec * SR)
                        gain = vel / 127.0
                        end = start + len(sample)
                        if end <= len(audio):
                            audio[start:end] += sample * gain
                        else:
                            # Wrap tail into the buffer (truncate if sample > buffer)
                            fit = len(audio) - start
                            audio[start:] += sample[:fit] * gain
                            overflow = min(len(sample) - fit, len(audio))
                            audio[:overflow] += sample[fit:fit + overflow] * gain

            time_offset += pat_dur

    # Soft-clip and normalize
    audio = np.tanh(audio * 1.5) * 0.85
    return audio


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------

def get_key():
    """Non-blocking key read. Returns None if no key available.

    Arrow keys are returned as 'up', 'down', 'left', 'right'.
    """
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Escape sequence — read remaining bytes if available
            if select.select([sys.stdin], [], [], 0.02)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[" and select.select([sys.stdin], [], [], 0.02)[0]:
                    ch3 = sys.stdin.read(1)
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch3)
            return None
        return ch
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def play_loop(buffers, labels, offsets, active, watch_path=None, reload_fn=None):
    """Loop audio with pattern selection (0-9), BPM control (w/s), and file watching.

    reload_fn: callable(bpm_delta) -> (buffers, labels, offsets) or (None, None, None)
    """
    cur = [active]
    pat_idx = [1]  # focused pattern (1-indexed), used for a/d navigation
    bpm_delta = [0]
    buf = [buffers[active].astype(np.float32)]
    pos = [0]

    def callback(outdata, frames, time_info, status):
        data = buf[0]
        length = len(data)
        p = pos[0] % length
        out = outdata[:, 0]
        remaining = length - p
        if remaining >= frames:
            out[:] = data[p:p + frames]
            pos[0] = p + frames
        else:
            out[:remaining] = data[p:]
            out[remaining:] = data[:frames - remaining]
            pos[0] = frames - remaining

    def swap(new_buffers, new_labels, new_offsets):
        buffers.update(new_buffers)
        labels.update(new_labels)
        offsets.update(new_offsets)
        sel = cur[0] if cur[0] in buffers else 0
        cur[0] = sel
        buf[0] = buffers[sel].astype(np.float32)
        pos[0] = 0
        return sel

    last_mtime = os.path.getmtime(watch_path) if watch_path else None

    with sd.OutputStream(samplerate=SR, channels=1, callback=callback):
        while True:
            time.sleep(0.05)

            key = get_key()
            if not key:
                pass

            # Pattern selection: 0-9
            elif key.isdigit():
                n = int(key)
                if n in buffers:
                    cur[0] = n
                    if n > 0:
                        pat_idx[0] = n
                    buf[0] = buffers[n].astype(np.float32)
                    pos[0] = 0
                    dur = len(buf[0]) / SR
                    print(f"  > [{n}] {labels[n]} ({dur:.1f}s)")

            # Previous/next pattern: a/d or left/right arrows
            elif key in ("a", "d", "left", "right"):
                n_pat = max(k for k in buffers if k > 0)
                step = 1 if key in ("d", "right") else -1
                pat_idx[0] = (pat_idx[0] - 1 + step) % n_pat + 1  # wrap 1..n_pat
                n = pat_idx[0]
                if cur[0] == 0:
                    # Seek within the "all" buffer
                    pos[0] = offsets.get(n, 0)
                else:
                    # Switch single-pattern loop
                    cur[0] = n
                    buf[0] = buffers[n].astype(np.float32)
                    pos[0] = 0
                dur = len(buffers[n]) / SR
                print(f"  > [{cur[0]}] {labels[n]} ({dur:.1f}s)")

            # BPM control: w/s or up/down arrows
            elif key in ("w", "s", "up", "down") and reload_fn:
                bpm_delta[0] += 10 if key in ("w", "up") else -10
                new_b, new_l, new_o = reload_fn(bpm_delta[0])
                if new_b:
                    sel = swap(new_b, new_l, new_o)
                    d = bpm_delta[0]
                    tag = f"BPM {d:+d}" if d else "BPM (original)"
                    print(f"  > {tag} — [{sel}] {labels[sel]}")

            # Check for file changes
            if watch_path and reload_fn:
                try:
                    mtime = os.path.getmtime(watch_path)
                    if mtime != last_mtime:
                        last_mtime = mtime
                        new_b, new_l, new_o = reload_fn(bpm_delta[0])
                        if new_b:
                            sel = swap(new_b, new_l, new_o)
                            print(f"  Reloaded — [{sel}] {labels[sel]}")
                except Exception as e:
                    print(f"  reload error: {e}", file=sys.stderr)


def save_wav(audio, path):
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_instruments():
    seen = {}
    for abbr, note in sorted(GM_DRUMS.items()):
        seen.setdefault(note, []).append(abbr)
    print("Available instruments:\n")
    for note in sorted(seen):
        print(f"  {'/'.join(seen[note]):>8s}  (MIDI {note})")


def build_buffers(patterns, samples):
    """Build per-pattern and combined audio buffers.

    Returns (buffers, labels, offsets) where offsets maps pattern key (1..N)
    to its start sample in the combined "all" buffer.
    """
    buffers = {}
    labels = {}
    offsets = {}

    # Individual patterns: keys 1..N
    sample_pos = 0
    for i, p in enumerate(patterns[:9]):
        audio = mix_patterns([p], samples)
        key = i + 1
        buffers[key] = audio
        offsets[key] = sample_pos
        sample_pos += len(audio)
        title = p["title"] or f"Pattern {key}"
        reps = p["repeat"] or 1
        bpm_str = f"{p['bpm']} BPM"
        labels[key] = f"{title} — {bpm_str}" + (f", {reps}x" if reps > 1 else "")

    # Combined: key 0
    buffers[0] = mix_patterns(patterns, samples)
    labels[0] = f"All ({len(patterns)} patterns)"

    return buffers, labels, offsets


def main():
    ap = argparse.ArgumentParser(
        description="Play drum patterns from text notation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Pattern format (in ```drums code blocks or plain text):

  BPM: 120
  Title: My Beat

  x-x-x-x-x-x-x-x- HH
  ----o-------o---   SD
  o-------o-o-----   BD

Step characters:
  x o   normal hit       X O   accent (loud)
  . g   ghost note       -     rest
""",
    )
    ap.add_argument("input", nargs="?", help="Input file (.md or plain text)")
    ap.add_argument("-s", "--save", metavar="FILE", help="Save to .wav file")
    ap.add_argument("-k", "--kit", metavar="DIR", nargs="?", const="", default="",
                    help="Load WAV samples from kit directory (default: kits/acoustic)")
    ap.add_argument("--synth", action="store_true", help="Use only synthesized sounds")
    ap.add_argument("-l", "--list", action="store_true", help="List available instruments")
    args = ap.parse_args()

    if args.list:
        list_instruments()
        return
    if not args.input:
        ap.print_help()
        sys.exit(1)

    if args.synth:
        samples = make_samples()
    else:
        kit_dir = args.kit or os.path.join(os.path.dirname(__file__), "kits", "acoustic")
        if os.path.isdir(kit_dir):
            samples = load_kit(kit_dir)
        else:
            if args.kit:
                print(f"Kit directory not found: {kit_dir}", file=sys.stderr)
                sys.exit(1)
            samples = make_samples()

    def load_file(bpm_delta=0):
        with open(args.input) as f:
            text = f.read()
        patterns = extract_patterns(text)
        if not patterns or all(not p["tracks"] for p in patterns):
            return None, None, None
        for p in patterns:
            p["bpm"] = max(20, p["bpm"] + bpm_delta)
        return build_buffers(patterns, samples)

    buffers, labels, offsets = load_file()
    if buffers is None:
        print("No drum patterns found.", file=sys.stderr)
        sys.exit(1)

    # Print pattern list
    n_patterns = len(buffers) - 1  # exclude key 0
    for key in range(1, n_patterns + 1):
        dur = len(buffers[key]) / SR
        print(f"  [{key}] {labels[key]} ({dur:.1f}s)")
    if n_patterns > 1:
        dur = len(buffers[0]) / SR
        print(f"  [0] {labels[0]} ({dur:.1f}s)")

    if args.save:
        save_wav(buffers[0], args.save)
        print(f"Saved {args.save}")

    # Start on the single pattern, or all if multiple
    active = 1 if n_patterns == 1 else 0
    dur = len(buffers[active]) / SR
    print(f"\nPlaying [{active}] {labels[active]} ({dur:.1f}s)")
    print(f"Watching {args.input} — 0-{min(n_patterns, 9)} switch, \u2190\u2192 prev/next, \u2191\u2193 BPM +/-10, Ctrl+C stop")

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        play_loop(buffers, labels, offsets, active, watch_path=args.input, reload_fn=load_file)
    except KeyboardInterrupt:
        sd.stop()
        print()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
