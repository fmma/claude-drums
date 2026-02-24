# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A real-time drum pattern player. Text-based drum notation in markdown files → audio playback with live editing. Single-file Python codebase (`drums.py`).

## Commands

```bash
pip install -r requirements.txt          # numpy, sounddevice
python drums.py patterns/basic-rock.md   # play a pattern
python drums.py patterns/basic-rock.md --kit kits/drumthrash  # with WAV kit
python drums.py patterns/basic-rock.md --synth                # synth only
python drums.py patterns/basic-rock.md --save out.wav         # export
python drums.py --list                   # list instruments
```

No tests, no linter, no build step.

## Architecture

Everything is in `drums.py`. The pipeline flows top-to-bottom:

1. **Macro expansion** (`apply_macros` → `process_block_macros`) — Expands `[norm]`, `[zoom]`, `[pat]`, `[rand]`, `[init]`, `[linear]` macros in drums blocks, writes results back to the file (macros are consumed on load).

2. **Parsing** (`parse_file` → `parse_preamble` + `parse_pattern_block`) — Extracts title, BPM, beats per bar, arrangement (list of lists of pattern names), and pattern dicts with tracks (instrument + step string).

3. **Sample loading** (`make_samples` / `load_kit`) — Synth engine generates sounds via numpy (FFT bandpass, pitch sweeps, noise). WAV kits override synth per-instrument. Variant samples `(note, "ghost")` and `(note, "accent")` provide articulation-specific sounds.

4. **Mixing** (`mix_patterns`) — For each pattern/track/step: look up sample by MIDI note, apply velocity as gain, sum into audio buffer. Ghost/accent characters select variant samples when available.

5. **Buffer building** (`build_buffers`) — Pre-renders pattern buffers, line buffers (concatenated patterns per arrangement line), and the full arrangement buffer. Returns a `song` dict used by playback.

6. **Playback** (`play_loop`) — `sounddevice.OutputStream` with callback. Three modes: SONG/LINE/PAT. Keyboard controls (space, m, arrows, k). File watcher triggers reload on edit.

## Key Data Structures

**Samples dict**: `{midi_note: np.array, (midi_note, "ghost"): np.array, ...}` — integer keys for normal hits, tuple keys for variants.

**Pattern dict**: `{"name": "VERSE", "tracks": [{"instrument": "HH", "steps": "x-x-x-x-"}], "steps_per_bar": 16, "bpm": 120, "beats": 4}`

**Song dict** (from `build_buffers`): Contains `arrangement_buffer`, `line_buffers`, `pattern_buffers`, `line_offsets`, `arrangement`, `pattern_names`, `title`, `bpm`.

## Key Constants

- `SR = 44100` — sample rate used everywhere
- `GM_DRUMS` — maps instrument abbreviations (BD, SD, HH, etc.) to MIDI note numbers
- Step chars: `x` = normal (vel 90), `o` = open hit (vel 90, HH → open hi-hat), `a` = accent (vel 110), `g` = ghost (vel 30), `f` = flam (vel 90), `-` = rest, `:` = rest (beat marker)

## Pattern File Format

```markdown
# Title
BPM 120
Verse Verse Chorus Fill

​```drums
Verse
x-x-x-x-x-x-x-x- HH
----x-------x--- SD
x-------x-x----- BD
​```
```

Preamble has title, BPM, BEATS (default 4), arrangement lines. Drums blocks have a name line then `steps INSTRUMENT` lines. Track length sets grid resolution (16 = 16th notes, 12 = triplets, 8 = 8th notes).

## WAV Kit Structure

Directory with `INSTRUMENT.wav` files (e.g. `BD.wav`, `SD.wav`). Variant files: `SD_ghost.wav`, `SD_accent.wav`. Supports 16/24/32-bit at any sample rate (auto-resampled to 44.1kHz). Four kits included: `acoustic` (GSCW), `drumthrash`, `tr-808`, `tr-909`.

## Macro System

Macros are case-insensitive, canonical form is lowercase. They are inline text substitutions — `[pat x-]` becomes `x-x-x-x-...`, `[init 16]` becomes `----------------`. Block macros `[norm]` and `[zoom N]` apply post-processing to all tracks. `[linear A B]` is a standalone line that expands to multiple track lines. All macros are consumed (written back to file).
