# claude-drums

Write drum patterns in markdown, hear them instantly. Loops indefinitely — Ctrl+C to stop.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python drums.py patterns/basic-rock.md
python drums.py patterns/shuffle.md
python drums.py patterns/fill-to-crash.md
python drums.py patterns/funky-groove.md --save groove.wav
python drums.py patterns/basic-rock.md --kit kits/acoustic
python drums.py patterns/basic-rock.md --kit kits/drumthrash
python drums.py patterns/four-on-the-floor.md --kit kits/tr-909
python drums.py patterns/hip-hop.md --kit kits/tr-808
python drums.py patterns/                                    # directory mode
python drums.py --list
```

## Example patterns

```
patterns/
  basic-rock.md
  four-on-the-floor.md
  funky-groove.md
  hip-hop.md
  bossa-nova.md
  shuffle.md
  fill-to-crash.md
  in-the-air-tonight.md
  smells-like-teen-spirit.md
  tongue-is-murder.md
```

## Sample kits

Use `--kit DIR` to replace the built-in synth sounds with WAV samples:

```bash
python drums.py patterns/basic-rock.md --kit kits/acoustic
```

A kit directory contains WAV files named by instrument abbreviation (`BD.wav`, `SD.wav`, `HH.wav`, etc.). Any missing instruments fall back to the built-in synth. Supports 16-bit, 24-bit, and 32-bit WAV files at any sample rate.

```
kits/
  acoustic/       — GSCW acoustic kit (24-bit, 48kHz)
    BD.wav          Kick drum          ← Kick-V05-Yamaha-16x16
    SD.wav          Snare              ← SNARE-V10-CustomWorks-6x13
    SD_ghost.wav    Snare ghost note   ← SNARE-V01-CustomWorks-6x13
    SD_accent.wav   Snare accent       ← SNARE-V14-CustomWorks-6x13
    HH.wav          Closed hi-hat      ← HHats-CL-V05-SABIAN-AAX
    OH.wav          Open hi-hat        ← HHats-OP-V04-SABIAN-AAX
    PH.wav          Pedal hi-hat       ← HHats-PDL-V02-SABIAN-AAX
    CR.wav          Crash 18"          ← 18-Crash-V03-SABIAN-18
    C2.wav          Crash 14"          ← 14-Crash-V03-SABIAN-14
    RD.wav          Ride 22"           ← Ride-V04-ROBMOR-SABIAN-22
    RB.wav          Ride bell          ← BELL-V04-ROBMOR-SABIAN-22
    SP.wav          Splash 6"          ← 6-Splash-V03-SABIAN-HH-6
    T1.wav          High tom 10"       ← V05-TTom 10 (Kit 2)
    T2.wav          Mid tom 12"        ← V05-TTom-12 (Kit 2)
    T3.wav          Floor tom 13"      ← V05-TTom13 (Kit 2)
    RS.wav          Rimshot            ← RIMSHOTS-V04-CW-6x13
    CL.wav          Sidestick          ← SSTICK-V04-CW-6x13
  tr-909/         — Roland TR-909 (16-bit, 44.1kHz)
    BD.wav          Bass drum          ← BT3A0D7
    SD.wav          Snare              ← ST7T7S7
    SD_ghost.wav    Snare ghost        ← ST7T7S3 (low snappy)
    SD_accent.wav   Snare accent       ← STAT7SA (high tune + snappy)
    HH.wav          Closed hi-hat      ← HHCD4
    OH.wav          Open hi-hat        ← HHOD4
    CR.wav          Crash cymbal       ← CSHD4
    RD.wav          Ride cymbal        ← RIDED4
    RS.wav          Rimshot            ← RIM127
    CL.wav          Handclap           ← HANDCLP1
    T1.wav          High tom           ← HT7D7
    T2.wav          Mid tom            ← MT7D7
    T3.wav          Low tom            ← LT7D7
  tr-808/         — Roland TR-808 (16-bit, 44.1kHz)
    BD.wav          Bass drum          ← BD0075
    SD.wav          Snare              ← SD5050
    SD_ghost.wav    Snare ghost        ← SD2525 (low tone + snap)
    SD_accent.wav   Snare accent       ← SD7575 (high tone + snap)
    HH.wav          Closed hi-hat      ← CH
    OH.wav          Open hi-hat        ← OH50
    PH.wav          Hi-hat accent      ← HC75
    CR.wav          Cymbal             ← CY0075
    RS.wav          Rimshot            ← RS
    CL.wav          Handclap           ← CP
    CB.wav          Cowbell            ← CB
    TM.wav          Maracas            ← MA
    T1.wav          High tom           ← HT75
    T2.wav          Mid tom            ← MT75
    T3.wav          Low tom            ← LT75
  drumthrash/     — DrumThrash acoustic kit (24-bit, 48kHz, Natural mic)
    BD.wav          Kick               ← KickA-Med-008
    SD.wav          Snare              ← SnareA-Med-005
    SD_ghost.wav    Snare ghost        ← SnareA-Soft-004
    SD_accent.wav   Snare accent       ← SnareA-Hard-008
    HH.wav          Closed hi-hat      ← Hat-Closed-004
    OH.wav          Open hi-hat        ← Hat-OpenMed-003
    PH.wav          Pedal hi-hat       ← Hat-Pedal-004
    CR.wav          Crash A            ← CrashA-003
    C2.wav          Crash B            ← CrashB-003
    RD.wav          Ride               ← Ride-003
    RB.wav          Ride bell          ← RideBell-005
    RS.wav          Rimshot            ← SnareA-Rimshot-006
    CL.wav          Side stick         ← Side-Stick-004
    T1.wav          Tom 1              ← Tom1-006
    T2.wav          Tom 2              ← Tom2-006
    T3.wav          Tom 4 (floor)      ← Tom4-006
```

Variant samples (`_ghost.wav`, `_accent.wav`) override the sound for ghost notes and accents on that instrument. If missing, the normal sample is used with velocity scaling.

### Sources

- **acoustic** — [GSCW Drum Kit Library](https://github.com/gregharvey/drum-samples) by Salvador Pelaez (G&S Custom Work), freeware. Kit 1 for most instruments, Kit 2 for toms.
- **drumthrash** — [Free Acoustic Drum Samples](https://www.drumthrash.com/free-drum-samples.html) by DrumThrash, royalty-free 24-bit/48kHz multi-sampled kit.
- **tr-909** — [Roland TR-909 sample set](https://www.drumkito.com/sample-packs/roland-tr-909-sample-pack/) by Rob Roy / Rob Roy Recordings (1995), free to copy and distribute.
- **tr-808** — [Roland TR-808 sample pack](https://www.drumkito.com/sample-packs/roland-tr-808-sample-pack/) from Drumkito.com, free download.

## Pattern Format

A file has an optional preamble (title, BPM, beats, arrangement) followed by named drums blocks:

```markdown
# Song Title

BPM 120

Verse Verse Chorus
Verse Verse Chorus Fill

```drums
Verse

x-x-x-x-x-x-x-x- HH
----x-------x--- SD
x-------x-x----- BD
```​

```drums
Chorus

x--------------- CR
x-x-x-x-x-x-x-x- HH
----x-------x--- SD
x---x---x---x--- BD
```​
```

If no arrangement is given, patterns play in file order. If no BPM is given, defaults to 120.

### Step characters

| Char | Meaning              | Velocity |
|------|----------------------|----------|
| `x`  | Normal hit           | 90       |
| `o`  | Open hit (HH → open) | 90       |
| `r`  | Rimshot (SD → rimshot)| 90       |
| `s`  | Sidestick (SD only)   | 90       |
| `b`  | Bell (RD → ride bell) | 90       |
| `a`  | Accent               | 110      |
| `g`  | Ghost note           | 30       |
| `f`  | Flam (grace note)    | 90       |
| `-`  | Rest                 | —        |
| `:`  | Rest (beat marker)   | —        |

### Subdivision

The number of characters per bar sets the grid resolution:

- **16 chars** → 16th notes (most common)
- **8 chars** → 8th notes
- **12 chars** → triplet 8th notes (shuffles)

### Settings

`BPM 120` in the preamble sets tempo (default 120).

`BEATS 5` sets beats per bar (default 4). A 20-step track with `BEATS 5` gives 4 steps per beat (16th notes in 5/4 time). Can also be set per-pattern inside a drums block to mix meters.

`NONORM` disables automatic normalization (see below).

## Macros

Macros are processed on load and written back to the file (consumed). They help generate and normalize patterns.

### Auto-formatting on save

`[fix]` and `[norm]` run automatically on every save (after all other macros):

1. **fix** — Pads tracks to the next length divisible by beats
2. **norm** — Pads tracks to equal length, removes evenly-spaced empty columns, sorts by MIDI note descending, marks beat positions with `:`

Add `NONORM` to the preamble to disable both.

### Block-level macros (standalone lines inside a drums block)

| Macro        | Description |
|--------------|-------------|
| `[norm]`     | Explicit norm (runs automatically unless `NONORM` is set) |
| `[fix]`      | Explicit fix (runs automatically unless `NONORM` is set) |
| `[dup]`      | Duplicate each track (doubles the bar length) |
| `[zoom N]`   | Intercalate N dashes after each step. `oooo` → `o--o--o--o--` with `[zoom 2]` |

### Inline macros (on a track line, replaces step data)

| Macro              | Description |
|--------------------|-------------|
| `[pat x-]`         | Repeat pattern to fill max track length. `x-` → `x-x-x-x-...` |
| `[rand N]`         | Random pattern with ~1/N hit probability per tick |
| `[init N]`         | Insert N dashes (empty steps) |
| `[linear SD BD]`   | Generate tracks where each tick has exactly one of the listed instruments |

## Controls

| Key          | Action                        |
|--------------|-------------------------------|
| `space`      | Pause / resume                |
| `m`          | Cycle mode: SONG → LINE → PAT |
| `left` / `right` or `a` / `d` | Navigate lines or patterns |
| `up` / `down` or `w` / `s`    | Adjust BPM ±10              |
| `k`          | Cycle sample kit              |
| `Ctrl+C`     | Stop                          |

File changes are auto-reloaded during playback.

### Directory mode

Pass a directory instead of a file to watch all `.md` files:

```bash
python drums.py patterns/
```

Plays the most recently saved pattern file. When you save a different file, it auto-switches. Works well with VS Code — edit and save pattern files and the player follows.

## Instruments

| Abbrev     | Sound           |
|------------|-----------------|
| BD / KD    | Bass Drum       |
| SD         | Snare (`r` rimshot, `s` sidestick) |
| HH / CH   | Hi-Hat          |
| PH         | Pedal Hi-Hat    |
| C1 / CR    | Crash 1         |
| C2         | Crash 2         |
| RD         | Ride (`b` for bell) |
| HT / T1   | High Tom        |
| MT / T2   | Mid Tom         |
| LT / FT / T3 | Low/Floor Tom |
| CB         | Cowbell         |
| TM         | Tambourine      |
| SP         | Splash Cymbal   |
