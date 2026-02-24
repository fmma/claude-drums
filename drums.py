#!/usr/bin/env python3
"""Generate and play drum patterns from text notation."""

import os
import sys
import re
import random
import tty
import time
import wave
import select
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


def _snare_ghost():
    """Soft center-of-head hit: more body, less wire, muted attack."""
    t = _t(0.2)
    freq1 = 180 * np.exp(-t * 8) + 140
    body = np.sin(2 * np.pi * np.cumsum(freq1) / SR) * np.exp(-t * 18) * 0.5
    noise = np.random.randn(len(t))
    wires = _bandpass(noise, 2000, 6000) * np.exp(-t * 20) * 0.3
    transient = np.random.randn(len(t)) * np.exp(-t * 80) * 0.2
    out = body + wires + transient
    return np.tanh(out) * 0.5


def _snare_accent():
    """Hard snare hit: more body, more wire, stronger transient."""
    t = _t(0.4)
    # Drum body: hit harder so more resonance and sustain
    freq1 = 190 * np.exp(-t * 5) + 145
    body1 = np.sin(2 * np.pi * np.cumsum(freq1) / SR) * np.exp(-t * 9) * 0.55
    body2 = np.sin(2 * np.pi * 320 * t) * np.exp(-t * 14) * 0.3
    # More wire sizzle from harder hit
    noise = np.random.randn(len(t))
    wires = _bandpass(noise, 2000, 9000) * np.exp(-t * 8) * 1.5
    # Harder attack transient
    transient = np.random.randn(len(t)) * np.exp(-t * 50) * 0.8
    out = body1 + body2 + wires + transient
    return np.tanh(out * 1.2) * 0.9


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
        (38, "ghost"): _snare_ghost(),
        (38, "accent"): _snare_accent(),
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
    # OH.wav still loads into note 46 (open hi-hat) even though OH is not a track instrument
    abbrev_to_note["OH"] = 46

    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(fname)[0].upper()
        # Support variant samples: SD_GHOST.wav, SD_ACCENT.wav
        variant = None
        for suffix in ("_GHOST", "_ACCENT"):
            if stem.endswith(suffix):
                variant = suffix[1:].lower()
                stem = stem[:-len(suffix)]
                break
        abbr = stem
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

            note = abbrev_to_note[abbr]
            if variant:
                samples[(note, variant)] = data
            else:
                samples[note] = data

        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}", file=sys.stderr)

    return samples


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

def get_velocity(char):
    if char in ("x", "o", "f"):
        return 90
    if char == "a":
        return 110
    if char == "g":
        return 30
    return 0


def parse_pattern_block(text):
    lines = text.strip().split("\n")
    name = ""
    tracks = []
    steps_per_bar = None
    beats = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        m = re.match(r"^([xoagf:\-]+)\s+([A-Za-z]\w{1,2})\s*$", stripped)
        if m:
            instrument = m.group(2).upper()
            steps_per_bar = max(steps_per_bar or 0, len(m.group(1)))
            tracks.append({"instrument": instrument, "steps": m.group(1)})
            continue

        m = re.match(r"BEATS\s+(\d+)", stripped, re.IGNORECASE)
        if m:
            beats = int(m.group(1))
            continue

        if not name and re.match(r'^[A-Za-z]\w*$', stripped):
            name = stripped

    return {"name": name, "tracks": tracks, "steps_per_bar": steps_per_bar or 16, "beats": beats}


def parse_preamble(text):
    """Extract title, global BPM, beats, and arrangement lines from preamble text."""
    title = ""
    global_bpm = None
    global_beats = None
    arrangement = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            continue
        m = re.match(r"BPM\s+(\d+)", stripped, re.IGNORECASE)
        if m:
            global_bpm = int(m.group(1))
            continue
        m = re.match(r"BEATS\s+(\d+)", stripped, re.IGNORECASE)
        if m:
            global_beats = int(m.group(1))
            continue
        if re.match(r"NONORM$", stripped, re.IGNORECASE):
            continue
        tokens = stripped.split()
        if all(re.match(r'^[A-Za-z]\w*$', t) for t in tokens):
            arrangement.append(tokens)
    return title, global_bpm, global_beats, arrangement


def process_block_macros(block_text, default_beats=None):
    """Process macros in a drums block and return (new_text, changed).

    Supported macros:
      [pat xxx]    - Tile pattern to fill remaining track length
      [rand N]     - Random hits with probability 1/N
      [init N]     - Insert N dashes (initialize empty track)
      [linear A B] - Distribute random hits across instruments (one line)
      [zoom N]     - Stretch all tracks by inserting N dashes after each step
      [dup]        - Duplicate each track (double the bar length)
      [fix]        - Pad tracks to next length divisible by beats
      [norm]       - Normalize: shrink to shortest grid, sort by pitch, mark beats with ':'
    """
    lines = block_text.split("\n")

    # Regex patterns for each macro and for track lines
    TRACK_RE = re.compile(r'^([xoagf:\-]+)\s+([A-Za-z]\w{1,2})\s*$')
    PAT_RE = re.compile(r'\[pat\s+([xoagf:\-]+)\]', re.IGNORECASE)
    RAND_RE = re.compile(r'\[rand\s+(\d+)\]', re.IGNORECASE)
    LINEAR_RE = re.compile(r'^\[linear\s+((?:[A-Za-z]\w{1,2}\s+)*[A-Za-z]\w{1,2})\]\s*$', re.IGNORECASE)
    INIT_RE = re.compile(r'\[init\s+(\d+)\]', re.IGNORECASE)
    ZOOM_RE = re.compile(r'\[zoom(?:\s+(\d+))?\]', re.IGNORECASE)
    NORM_RE = re.compile(r'\[norm\]', re.IGNORECASE)
    FIX_RE = re.compile(r'\[fix\]', re.IGNORECASE)
    DUP_RE = re.compile(r'\[dup\]', re.IGNORECASE)

    BEATS_RE = re.compile(r'BEATS\s+(\d+)', re.IGNORECASE)

    # --- First pass: detect which macros are present ---
    zoom_factor = None
    has_norm = False
    has_fix = False
    has_dup = False
    has_macro = False
    block_beats = default_beats or 4

    for line in lines:
        stripped = line.strip()
        if PAT_RE.search(stripped) or RAND_RE.search(stripped) or INIT_RE.search(stripped):
            has_macro = True
        if LINEAR_RE.match(stripped):
            has_macro = True
        zm = ZOOM_RE.search(stripped)
        if zm:
            has_macro = True
            zoom_factor = int(zm.group(1) or 1)
        if NORM_RE.search(stripped):
            has_macro = True
            has_norm = True
        if FIX_RE.search(stripped):
            has_macro = True
            has_fix = True
        if DUP_RE.search(stripped):
            has_macro = True
            has_dup = True
        bm = BEATS_RE.match(stripped)
        if bm:
            block_beats = int(bm.group(1))

    if not has_macro:
        return (block_text, False)

    # --- Compute max track length (used by PAT and RAND to fill remaining space) ---
    max_len = 0
    for line in lines:
        tm = TRACK_RE.match(line.strip())
        if tm:
            max_len = max(max_len, len(tm.group(1)))

    # --- Second pass: expand inline macros line by line ---
    target_len = max_len or 16

    def expand_pat(m):
        """Tile the pattern to fill from the macro's position to target_len."""
        pat = m.group(1)
        length = max(target_len - m.start(), 0)
        if not length:
            return ''
        return (pat * ((length // len(pat)) + 1))[:length]

    def expand_rand(m):
        """Generate random hits with probability 1/N."""
        n = int(m.group(1))
        length = max(target_len - m.start(), 0)
        if not length:
            return ''
        return ''.join('x' if random.random() < 1.0 / n else '-' for _ in range(length))

    result = []
    for line in lines:
        stripped = line.strip()

        # LINEAR is a block-level macro: one line becomes multiple track lines
        lm = LINEAR_RE.match(stripped)
        if lm:
            instruments = lm.group(1).split()
            choices = [random.choice(instruments) for _ in range(target_len)]
            for inst in instruments:
                steps = ''.join('x' if choices[i] == inst else '-' for i in range(target_len))
                result.append(f"{steps} {inst}")
            continue

        # Strip tag-only macros, expand generative macros
        # Strip previous warnings (will be regenerated if needed)
        if stripped.startswith("# WARNING:"):
            continue
        line = ZOOM_RE.sub('', line)
        line = NORM_RE.sub('', line)
        line = FIX_RE.sub('', line)
        line = DUP_RE.sub('', line)
        line = INIT_RE.sub(lambda m: '-' * int(m.group(1)), line)
        line = PAT_RE.sub(expand_pat, line)
        line = RAND_RE.sub(expand_rand, line)
        result.append(line)

    # --- Post-processing: DUP, FIX, ZOOM and NORM transform entire tracks ---
    if zoom_factor is not None or has_norm or has_fix or has_dup:
        # Collect track lines with their positions in the result
        track_entries = []
        for i, line in enumerate(result):
            tm = TRACK_RE.match(line.strip())
            if tm:
                track_entries.append((i, tm.group(1), tm.group(2)))

        if track_entries:
            tracks = [(steps, inst) for _, steps, inst in track_entries]

            # DUP: double each track by concatenating with itself
            if has_dup:
                tracks = [(steps + steps, inst) for steps, inst in tracks]

            # FIX: pad tracks to next length divisible by beats
            if has_fix:
                max_len = max(len(steps) for steps, _ in tracks)
                if max_len % block_beats != 0:
                    max_len += block_beats - (max_len % block_beats)
                tracks = [(steps.ljust(max_len, '-'), inst) for steps, inst in tracks]

            # ZOOM: insert dashes after each step character
            if zoom_factor is not None:
                tracks = [(''.join(ch + '-' * zoom_factor for ch in steps), inst)
                          for steps, inst in tracks]

            # NORM: pad to equal length, shrink common empty positions, sort by pitch
            if has_norm:
                tracks, warning = _normalize_tracks(tracks, beats=block_beats)

            # Write transformed tracks back into result lines
            if has_norm:
                # NORM reorders tracks, so gather them at the first track position
                first_pos = track_entries[0][0]
                for j in range(len(track_entries) - 1, -1, -1):
                    del result[track_entries[j][0]]
                for j, (steps, inst) in enumerate(tracks):
                    result.insert(first_pos + j, f"{steps} {inst}")
                if warning:
                    result.insert(first_pos, warning)
            else:
                for j, (idx, _, _) in enumerate(track_entries):
                    result[idx] = f"{tracks[j][0]} {tracks[j][1]}"

    return ('\n'.join(result), True)


def _normalize_tracks(tracks, beats=4):
    """Pad tracks to equal length, remove common empty grid positions, sort by pitch.

    This is the NORM post-processing step. It finds the smallest factor p such
    that all non-zero positions in every track fall on multiples of p, then keeps
    only those positions (shrinking the grid). Repeats until no further shrinking
    is possible. Rest characters are then set to ':' on beat positions and '-'
    elsewhere.
    """
    # Pad all tracks to the same length
    max_len = max(len(steps) for steps, _ in tracks)
    tracks = [(steps.ljust(max_len, '-'), inst) for steps, inst in tracks]

    # Iteratively try to shrink by factor p (smallest first)
    # Never shrink below beats (1 step per beat is the minimum grid)
    min_len = max(beats, 1)
    n = len(tracks[0][0])
    p = 2
    while p <= n and n // p >= min_len:
        if n % p == 0:
            # Check if all hits land on positions divisible by p
            can_shrink = all(
                steps[i] in ('-', ':')
                for steps, _ in tracks
                for i in range(n) if i % p != 0
            )
            if can_shrink:
                tracks = [(''.join(steps[i] for i in range(0, n, p)), inst)
                          for steps, inst in tracks]
                n = len(tracks[0][0])
                # Restart: a new smaller factor may apply after shrinking
            else:
                p += 1
        else:
            p += 1

    # Replace rest characters: ':' on beat positions, '-' elsewhere
    n = len(tracks[0][0])
    if beats > 0 and n % beats == 0:
        beat_step = n // beats
        def rest_char(i):
            return ':' if i % beat_step == 0 else '-'
        tracks = [
            (''.join(rest_char(i) if ch in ('-', ':') else ch for i, ch in enumerate(steps)), inst)
            for steps, inst in tracks
        ]
    elif beats > 0:
        warning = f"# WARNING: {n} ticks does not divide evenly by {beats} beats"
        return tracks, warning

    # Sort high-pitched instruments first (descending MIDI note)
    tracks.sort(key=lambda t: GM_DRUMS.get(t[1].upper(), 0), reverse=True)
    return tracks, None


def _norm_block(body, default_beats=None):
    """Apply [norm] to a drums block body. Returns (new_body, changed)."""
    TRACK_RE = re.compile(r'^([xoagf:\-]+)\s+([A-Za-z]\w{1,2})\s*$')
    BEATS_RE = re.compile(r'BEATS\s+(\d+)', re.IGNORECASE)

    beats = default_beats or 4
    lines = [l for l in body.split("\n") if not l.strip().startswith("# WARNING:")]
    for line in lines:
        bm = BEATS_RE.match(line.strip())
        if bm:
            beats = int(bm.group(1))

    track_entries = []
    for i, line in enumerate(lines):
        tm = TRACK_RE.match(line.strip())
        if tm:
            track_entries.append((i, tm.group(1), tm.group(2)))

    if not track_entries:
        return body, False

    tracks = [(steps, inst) for _, steps, inst in track_entries]

    # Fix: pad tracks to next length divisible by beats
    max_len = max(len(steps) for steps, _ in tracks)
    if max_len % beats != 0:
        max_len += beats - (max_len % beats)
    tracks = [(steps.ljust(max_len, '-'), inst) for steps, inst in tracks]

    tracks, warning = _normalize_tracks(tracks, beats=beats)

    first_pos = track_entries[0][0]
    for j in range(len(track_entries) - 1, -1, -1):
        del lines[track_entries[j][0]]
    for j, (steps, inst) in enumerate(tracks):
        lines.insert(first_pos + j, f"{steps} {inst}")
    if warning:
        lines.insert(first_pos, warning)

    new_body = '\n'.join(lines)
    return new_body, new_body != body


def apply_macros(text, filepath):
    """Expand macros in all ```drums blocks and write the result back to file.

    Returns the updated text. After all macros are expanded, [norm] runs
    automatically on every block. Use NONORM in the preamble to disable.
    """
    changed = False
    parts = re.split(r'(```drums\s*\n.*?```)', text, flags=re.DOTALL)

    # Scan preamble for settings
    global_norm = True
    global_beats = None
    for i in range(0, len(parts), 2):
        # Explicit global [norm] is now redundant — strip it silently
        if re.search(r'^\[norm\]\s*$', parts[i], re.MULTILINE | re.IGNORECASE):
            parts[i] = re.sub(r'^\[norm\]\s*\n?', '', parts[i], flags=re.MULTILINE | re.IGNORECASE)
            changed = True
        if re.search(r'^NONORM\s*$', parts[i], re.MULTILINE | re.IGNORECASE):
            global_norm = False
        bm = re.search(r'^BEATS\s+(\d+)', parts[i], re.MULTILINE | re.IGNORECASE)
        if bm:
            global_beats = int(bm.group(1))

    # Pass 1: expand all macros (pat, rand, linear, zoom, fix, dup, and explicit [norm])
    for i in range(1, len(parts), 2):
        m = re.match(r'```drums\s*\n(.*?)```', parts[i], re.DOTALL)
        if not m:
            continue
        new_body, block_changed = process_block_macros(m.group(1), default_beats=global_beats)
        if block_changed:
            changed = True
            parts[i] = f"```drums\n{new_body}```"

    # Pass 2: auto-norm on every block (unless NONORM)
    if global_norm:
        for i in range(1, len(parts), 2):
            m = re.match(r'```drums\s*\n(.*?)```', parts[i], re.DOTALL)
            if not m:
                continue
            new_body, norm_changed = _norm_block(m.group(1), default_beats=global_beats)
            if norm_changed:
                changed = True
                parts[i] = f"```drums\n{new_body}```"

    new_text = ''.join(parts)
    if changed:
        with open(filepath, 'w') as f:
            f.write(new_text)
    return new_text


def parse_file(text):
    """Parse a complete file with preamble arrangement and drums blocks."""
    preamble_split = re.split(r'```drums\s*\n', text, maxsplit=1)
    preamble_text = preamble_split[0] if len(preamble_split) > 1 else ""
    title, global_bpm, global_beats, arrangement = parse_preamble(preamble_text)

    blocks = re.findall(r"```drums\s*\n(.*?)```", text, re.DOTALL)
    bpm = global_bpm or 120
    beats = global_beats or 4
    parsed = [parse_pattern_block(b) for b in blocks]

    patterns = {}
    for i, p in enumerate(parsed):
        name = (p["name"] or f"PATTERN {i + 1}").upper()
        p["name"] = name
        p["bpm"] = bpm
        p["beats"] = p["beats"] or beats
        patterns[name] = p

    arrangement = [[t.upper() for t in line] for line in arrangement]
    if not arrangement:
        arrangement = [[name] for name in patterns]

    return {
        "title": title,
        "bpm": bpm,
        "arrangement": arrangement,
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def _place_sample(audio, start, smp, gain):
    """Add a sample into the audio buffer at the given position."""
    end = start + len(smp)
    if end <= len(audio):
        audio[start:end] += smp * gain
    else:
        fit = len(audio) - start
        audio[start:] += smp[:fit] * gain
        overflow = min(len(smp) - fit, len(audio))
        audio[:overflow] += smp[fit:fit + overflow] * gain


def mix_patterns(patterns, samples, loop=False):
    # Calculate total duration in seconds
    total_secs = 0.0
    for p in patterns:
        spb = p["steps_per_bar"]
        step_s = (60.0 / p["bpm"]) * (p["beats"] / spb)
        max_steps = max((len(t["steps"]) for t in p["tracks"]), default=0)
        total_secs += max_steps * step_s

    audio = np.zeros(int(SR * total_secs))
    hh_events = []  # (sample_pos, char, vel) — deferred for choke-aware pass
    time_offset = 0.0

    for p in patterns:
        spb = p["steps_per_bar"]
        step_s = (60.0 / p["bpm"]) * (p["beats"] / spb)
        max_steps = max((len(t["steps"]) for t in p["tracks"]), default=0)
        pat_dur = max_steps * step_s

        for t in p["tracks"]:
            inst = t["instrument"]
            if inst not in GM_DRUMS:
                print(f"Warning: unknown instrument '{inst}'", file=sys.stderr)
                continue
            note = GM_DRUMS[inst]
            if note not in samples:
                continue
            sample = samples[note]

            events = [(i, char, get_velocity(char))
                      for i, char in enumerate(t["steps"])
                      if get_velocity(char) > 0]

            # Defer HH (note 42) for choke-aware second pass
            if note == 42:
                for i, char, vel in events:
                    t_sec = time_offset + i * step_s
                    hh_events.append((int(t_sec * SR), char, vel))
                continue

            for ev_idx, (i, char, vel) in enumerate(events):
                if char == "g" and (note, "ghost") in samples:
                    smp = samples[(note, "ghost")]
                elif char == "a" and (note, "accent") in samples:
                    smp = samples[(note, "accent")]
                else:
                    smp = sample

                # Flam: place a quiet grace note ~20ms before the main hit
                if char == "f":
                    t_grace = time_offset + i * step_s - 0.020
                    if t_grace >= 0:
                        g_start = int(t_grace * SR)
                        _place_sample(audio, g_start, sample, vel / 127.0 * 0.4)

                t_sec = time_offset + i * step_s
                _place_sample(audio, int(t_sec * SR), smp, vel / 127.0)

        time_offset += pat_dur

    # Second pass: place HH events with choke awareness across all patterns
    hh_events.sort(key=lambda e: e[0])
    hh_sample = samples.get(42)
    open_hh_sample = samples.get(46)

    for ev_idx, (pos, char, vel) in enumerate(hh_events):
        if char == "o" and open_hh_sample is not None:
            smp = open_hh_sample
        elif char == "g" and (42, "ghost") in samples:
            smp = samples[(42, "ghost")]
        elif char == "a" and (42, "accent") in samples:
            smp = samples[(42, "accent")]
        elif hh_sample is not None:
            smp = hh_sample
        else:
            continue

        # Hi-hat choke: truncate open hi-hat at next HH hit
        if char == "o" and open_hh_sample is not None:
            next_pos = None
            if ev_idx + 1 < len(hh_events):
                next_pos = hh_events[ev_idx + 1][0]
            elif loop and len(hh_events) > 1:
                next_pos = len(audio) + hh_events[0][0]
            if next_pos is not None:
                choke_len = next_pos - pos
                if 0 < choke_len < len(smp):
                    smp = smp[:choke_len].copy()
                    fade = min(int(0.005 * SR), choke_len)
                    if fade > 0:
                        smp[-fade:] *= np.linspace(1, 0, fade)

        # Flam
        if char == "f" and hh_sample is not None:
            t_grace = pos / SR - 0.020
            if t_grace >= 0:
                _place_sample(audio, int(t_grace * SR), hh_sample, vel / 127.0 * 0.4)

        _place_sample(audio, pos, smp, vel / 127.0)

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

def play_loop(song, watch_fn=None, reload_fn=None, kit_names=None, switch_kit_fn=None, start_kit_idx=0):
    """Arrangement-based playback with SONG/LINE/PAT modes."""
    MODES = ["SONG", "LINE", "PAT"]
    mode = [0]
    line_idx = [0]
    pat_idx = [0]
    paused = [False]
    bpm_delta = [0]
    pos = [0]
    pat_names = [song["pattern_names"][:]]
    kit_list = kit_names or []
    kit_idx = [start_kit_idx % len(kit_list)] if kit_list else [0]

    def get_active_buf():
        m = MODES[mode[0]]
        if m == "SONG":
            return song["arrangement_buffer"]
        elif m == "LINE":
            idx = min(line_idx[0], len(song["line_buffers"]) - 1)
            return song["line_buffers"][idx]
        else:
            idx = min(pat_idx[0], len(pat_names[0]) - 1)
            return song["pattern_buffers"][pat_names[0][idx]]

    active_buf = [get_active_buf().astype(np.float32)]

    def switch_buf():
        active_buf[0] = get_active_buf().astype(np.float32)
        if MODES[mode[0]] == "SONG":
            offsets = song["line_offsets"]
            idx = min(line_idx[0], len(offsets) - 2)
            pos[0] = offsets[idx]
        else:
            pos[0] = 0

    def callback(outdata, frames, time_info, status):
        if paused[0]:
            outdata[:] = 0
            return
        data = active_buf[0]
        length = len(data)
        if length == 0:
            outdata[:] = 0
            return
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

    def current_line_from_pos():
        offsets = song["line_offsets"]
        p = pos[0]
        for i in range(len(offsets) - 1):
            if p < offsets[i + 1]:
                return i
        return max(0, len(offsets) - 2)

    def format_status():
        m = MODES[mode[0]]
        arr = song["arrangement"]
        n_lines = len(arr)
        pause_str = " [PAUSED]" if paused[0] else ""
        kit_str = f" [{kit_list[kit_idx[0]]}]" if kit_list else ""
        if m in ("SONG", "LINE"):
            li = min(line_idx[0], n_lines - 1)
            tokens = " ".join(arr[li])
            return f"\r\x1b[2K  {m:4s} {song['bpm']}bpm{kit_str} | Line {li+1}/{n_lines}: {tokens}{pause_str}"
        else:
            names = pat_names[0]
            idx = min(pat_idx[0], len(names) - 1)
            return f"\r\x1b[2K  {m:4s} {song['bpm']}bpm{kit_str} | {names[idx]} ({idx+1}/{len(names)}){pause_str}"

    def print_status():
        print(format_status(), end="", flush=True)

    def do_reload(new_song):
        song.update(new_song)
        pat_names[0] = song["pattern_names"][:]
        line_idx[0] = min(line_idx[0], len(song["arrangement"]) - 1)
        pat_idx[0] = min(pat_idx[0], len(pat_names[0]) - 1)
        switch_buf()

    prev_line = [-1]
    print_status()

    with sd.OutputStream(samplerate=SR, channels=1, callback=callback):
        while True:
            time.sleep(0.05)

            # Track current line in SONG mode
            if MODES[mode[0]] == "SONG" and not paused[0]:
                cl = current_line_from_pos()
                if cl != prev_line[0]:
                    prev_line[0] = cl
                    line_idx[0] = cl
                    print_status()

            key = get_key()
            if not key:
                pass

            elif key == " ":
                paused[0] = not paused[0]
                print_status()

            elif key == "m":
                mode[0] = (mode[0] + 1) % len(MODES)
                switch_buf()
                prev_line[0] = -1
                print_status()

            elif key in ("a", "d", "left", "right"):
                step = 1 if key in ("d", "right") else -1
                m = MODES[mode[0]]
                if m in ("SONG", "LINE"):
                    n = len(song["arrangement"])
                    line_idx[0] = (line_idx[0] + step) % n
                    prev_line[0] = line_idx[0]
                else:
                    n = len(pat_names[0])
                    pat_idx[0] = (pat_idx[0] + step) % n
                switch_buf()
                print_status()

            elif key in ("w", "s", "up", "down") and reload_fn:
                bpm_delta[0] += 10 if key in ("w", "up") else -10
                new_song = reload_fn(bpm_delta[0])
                if new_song:
                    do_reload(new_song)
                    print_status()

            elif key == "k" and kit_list and switch_kit_fn:
                kit_idx[0] = (kit_idx[0] + 1) % len(kit_list)
                new_song = switch_kit_fn(kit_list[kit_idx[0]], bpm_delta[0])
                if new_song:
                    do_reload(new_song)
                    print_status()

            # File watching
            if watch_fn and reload_fn:
                try:
                    if watch_fn():
                        new_song = reload_fn(bpm_delta[0])
                        if new_song:
                            do_reload(new_song)
                            print_status()
                except Exception as e:
                    print(f"\r  reload error: {e}", file=sys.stderr)


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


def build_buffers(file_data, samples):
    """Build audio buffers for arrangement-based playback."""
    patterns = file_data["patterns"]
    arrangement = file_data["arrangement"]

    # Per-pattern audio (for PAT mode — loop choke so open HH chokes on repeat)
    pattern_buffers = {}
    for name, p in patterns.items():
        pattern_buffers[name] = mix_patterns([p], samples, loop=True)

    # Per-line audio: mix patterns together (not just concatenate) so hi-hat
    # choke works across pattern boundaries within a line
    line_buffers = []
    for line_tokens in arrangement:
        line_pats = []
        for token in line_tokens:
            if token in patterns:
                line_pats.append(patterns[token])
            else:
                print(f"Warning: pattern '{token}' not found", file=sys.stderr)
        if line_pats:
            line_buffers.append(mix_patterns(line_pats, samples, loop=True))
        else:
            line_buffers.append(np.zeros(SR))

    # Arrangement buffer: mix all patterns together for cross-line choke
    all_pats = [patterns[t] for line in arrangement for t in line if t in patterns]
    arrangement_buffer = mix_patterns(all_pats, samples, loop=True) if all_pats else np.zeros(SR)

    # Line offsets (from line buffer lengths — same total duration as arrangement)
    line_offsets = []
    pos = 0
    for buf in line_buffers:
        line_offsets.append(pos)
        pos += len(buf)
    line_offsets.append(pos)  # sentinel

    return {
        "title": file_data["title"],
        "bpm": file_data["bpm"],
        "arrangement": arrangement,
        "pattern_names": list(patterns.keys()),
        "pattern_buffers": pattern_buffers,
        "line_buffers": line_buffers,
        "arrangement_buffer": arrangement_buffer,
        "line_offsets": line_offsets,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Play drum patterns from text notation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Pattern format (in ```drums code blocks or plain text):

  BPM 120

  x-x-x-x-x-x-x-x- HH
  ----x-------x---   SD
  x-------x-x-----   BD

Step characters:
  x     normal hit       a     accent (loud)
  o     open hit (HH)    g     ghost note
  f     flam             -     rest
""",
    )
    ap.add_argument("input", nargs="?", help="Input file or directory (dir mode: plays most recently saved .md)")
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

    # Discover available kits
    kits_dir = os.path.join(os.path.dirname(__file__), "kits")
    kit_names = ["synth"]
    if os.path.isdir(kits_dir):
        kit_names += sorted(d for d in os.listdir(kits_dir)
                            if os.path.isdir(os.path.join(kits_dir, d)))

    kit_cache = {}

    def get_samples(kit_name):
        if kit_name not in kit_cache:
            if kit_name == "synth":
                kit_cache[kit_name] = make_samples()
            else:
                kit_cache[kit_name] = load_kit(os.path.join(kits_dir, kit_name))
        return kit_cache[kit_name]

    if args.synth:
        current_kit = "synth"
    elif args.kit:
        kit_dir = args.kit
        if os.path.isdir(kit_dir):
            current_kit = os.path.basename(kit_dir)
            kit_cache[current_kit] = load_kit(kit_dir)
            if current_kit not in kit_names:
                kit_names.append(current_kit)
        else:
            print(f"Kit directory not found: {kit_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        current_kit = "acoustic" if "acoustic" in kit_names else kit_names[0]

    samples = get_samples(current_kit)
    kit_start_idx = kit_names.index(current_kit) if current_kit in kit_names else 0

    # Directory mode: watch all .md files, play the most recently saved one
    watch_dir = None
    current_file = [os.path.abspath(args.input)]

    if os.path.isdir(args.input):
        watch_dir = os.path.abspath(args.input)
        md_files = [f for f in os.listdir(watch_dir) if f.lower().endswith('.md')]
        if not md_files:
            print("No .md files found in directory", file=sys.stderr)
            sys.exit(1)
        md_files.sort(key=lambda f: os.path.getmtime(os.path.join(watch_dir, f)))
        current_file[0] = os.path.join(watch_dir, md_files[-1])

    def load_file(bpm_delta=0):
        if watch_dir:
            md_files = [f for f in os.listdir(watch_dir) if f.lower().endswith('.md')]
            if md_files:
                md_files.sort(key=lambda f: os.path.getmtime(os.path.join(watch_dir, f)))
                newest = os.path.join(watch_dir, md_files[-1])
                if newest != current_file[0]:
                    current_file[0] = newest
                    print(f"\r\x1b[2K  \u2192 {os.path.basename(newest)}", flush=True)
        with open(current_file[0]) as f:
            text = f.read()
        text = apply_macros(text, current_file[0])
        file_data = parse_file(text)
        if not file_data["patterns"] or all(
            not p["tracks"] for p in file_data["patterns"].values()
        ):
            return None
        for p in file_data["patterns"].values():
            p["bpm"] = max(20, p["bpm"] + bpm_delta)
        file_data["bpm"] = max(20, file_data["bpm"] + bpm_delta)
        return build_buffers(file_data, samples)

    def switch_kit(kit_name, bpm_delta=0):
        nonlocal samples
        samples = get_samples(kit_name)
        return load_file(bpm_delta)

    # Create watch function: checks if any watched file was modified
    if watch_dir:
        def _dir_max_mtime():
            max_mt = 0
            for f in os.listdir(watch_dir):
                if f.lower().endswith('.md'):
                    try:
                        max_mt = max(max_mt, os.path.getmtime(os.path.join(watch_dir, f)))
                    except OSError:
                        pass
            return max_mt
        _last_mt = [_dir_max_mtime()]
        def watch_fn():
            mt = _dir_max_mtime()
            if mt != _last_mt[0]:
                _last_mt[0] = mt
                return True
            return False
    else:
        _last_mt = [os.path.getmtime(current_file[0])]
        def watch_fn():
            try:
                mt = os.path.getmtime(current_file[0])
            except OSError:
                return False
            if mt != _last_mt[0]:
                _last_mt[0] = mt
                return True
            return False

    song = load_file()
    if song is None:
        print("No drum patterns found.", file=sys.stderr)
        sys.exit(1)

    if watch_dir:
        print(f"  Watching: {watch_dir}/")
    if song["title"]:
        print(f"  {os.path.basename(current_file[0])}  {song['title']}" if watch_dir else f"  {song['title']}")
    print(f"  BPM {song['bpm']}  Kit: {current_kit}")
    print(f"  Kits: {', '.join(kit_names)}")
    print()
    for i, line_tokens in enumerate(song["arrangement"]):
        dur = len(song["line_buffers"][i]) / SR
        print(f"  {i+1:2d}. {' '.join(line_tokens)} ({dur:.1f}s)")
    print()
    print(f"  Patterns: {', '.join(song['pattern_names'])}")

    if args.save:
        save_wav(song["arrangement_buffer"], args.save)
        print(f"  Saved {args.save}")

    dur = len(song["arrangement_buffer"]) / SR
    print(f"\n  Playing ({dur:.1f}s total)")
    print(f"  space pause, m mode, arrows navigate, up/down BPM, k kit, Ctrl+C stop")

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        play_loop(song, watch_fn=watch_fn, reload_fn=load_file,
                  kit_names=kit_names, switch_kit_fn=switch_kit,
                  start_kit_idx=kit_start_idx)
    except KeyboardInterrupt:
        sd.stop()
        print()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
