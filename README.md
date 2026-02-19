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
    BD.wav          Kick drum (Yamaha 16x16)
    SD.wav          Snare (CustomWorks 6x13)
    HH.wav          Closed hi-hat (Sabian AAX)
    OH.wav          Open hi-hat
    PH.wav          Pedal hi-hat
    CR.wav          Crash 18" (Sabian)
    C2.wav          Crash 14" (Sabian)
    RD.wav          Ride 22" (Sabian)
    RB.wav          Ride bell
    SP.wav          Splash 6" (Sabian HH)
    T1.wav          High tom 10" (StarClassic)
    T2.wav          Mid tom 12"
    T3.wav          Floor tom 13" (StarClassic)
    RS.wav          Rimshot
    CL.wav          Sidestick
```

## Pattern Format

Write patterns in `` ```drums `` code blocks inside any `.md` file (or as plain text).
Multiple blocks in one file play sequentially — chain a groove into a fill.

```drums
BPM: 120
Title: Pattern Name

x-x-x-x-x-x-x-x- HH
----o-------o---   SD
o-------o-o-----   BD
```

### Step characters

| Char    | Meaning      | Velocity |
|---------|--------------|----------|
| `x` `o` | Normal hit  | 100      |
| `X` `O` | Accent      | 127      |
| `.` `g` | Ghost note  | 50       |
| `-`     | Rest         | —        |

### Subdivision

The number of characters per bar sets the grid resolution:

- **16 chars** → 16th notes (most common)
- **8 chars** → 8th notes
- **12 chars** → triplet 8th notes (shuffles)

### Settings

| Setting      | Default | Description                   |
|--------------|---------|-------------------------------|
| `BPM: 120`  | 120     | Tempo                         |
| `Swing: 50` | 0       | Delays off-beat steps (0–100) |

## Instruments

| Abbrev     | Sound           |
|------------|-----------------|
| BD / KD    | Bass Drum       |
| SD         | Snare           |
| RS         | Rimshot         |
| CL         | Clap            |
| HH / CH   | Closed Hi-Hat   |
| OH         | Open Hi-Hat     |
| PH         | Pedal Hi-Hat    |
| CR         | Crash Cymbal    |
| C2         | Crash 2         |
| RD         | Ride Cymbal     |
| RB         | Ride Bell       |
| HT         | High Tom        |
| MT         | Mid Tom         |
| LT / FT   | Low/Floor Tom   |
| CB         | Cowbell         |
| TM         | Tambourine      |
| SP         | Splash Cymbal   |
