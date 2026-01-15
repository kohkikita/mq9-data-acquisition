# STM32 Load Cell + Event-Triggered Acoustic Analysis Logger

## Overview

This project implements an **event-triggered data acquisition and analysis pipeline** for propeller testing.  
It synchronously captures:

- **Thrust / force data** from an STM32-based load cell system (via USB serial)
- **Audio data** from a USB microphone
- **Time-aligned post-processed metrics**, including:
  - Force vs time
  - Audio RMS level vs time
  - Time–frequency spectrograms (STFT)
  - CSV datasets for offline analysis

Only audio associated with **force events** is saved and analyzed, reducing data volume and focusing analysis on physically meaningful operating regions.

---

## System Architecture

```
STM32 Load Cell  ── USB Serial ──┐
                                 ├── Python DAQ + GUI
USB Microphone ── Audio API  ────┘
                                      ↓
                           Event-Triggered Logging
                                      ↓
                           Post-Processing & Plots
```

---

## Key Features

### Event-Triggered Capture

- Recording begins only after force exceeds a configurable threshold
- Includes **pre-event** and **post-event** samples
- Automatically stops once force falls below threshold for a defined duration

### Synchronous Force + Audio

- Force and audio are captured concurrently
- Audio is trimmed to include **only event-relevant data**
- Post-processing aligns audio features to force timestamps

### Post-Processing Outputs

For each run, the system produces:

| File | Description |
|-----|-------------|
| `*.csv` | Raw force event log |
| `*.wav` | Event-only audio |
| `*_combined_event_aligned.csv` | Force + audio RMS aligned in time |
| `*_force_audio_plot.png` | Dual-axis force vs audio RMS plot |
| `*_audio_spectrogram.png` | Time–frequency spectrogram (STFT) |
| `*_audio_spectrogram.csv` | Time-resolved frequency magnitude data |

---

## Output Directory Structure

```
runs/
├── runname_YYYY-MM-DD_HH-MM-SS.csv
├── runname_YYYY-MM-DD_HH-MM-SS.wav
├── runname_YYYY-MM-DD_HH-MM-SS_combined_event_aligned.csv
├── runname_YYYY-MM-DD_HH-MM-SS_force_audio_plot.png
├── runname_YYYY-MM-DD_HH-MM-SS_audio_spectrogram.png
└── runname_YYYY-MM-DD_HH-MM-SS_audio_spectrogram.csv
```

Run names are user-defined and automatically suffixed with date and time.

---

## How the Data Is Calculated

### Force Data

- Parsed from STM32 serial output:
  ```
  Load=0.123 N, t=4567 ms
  ```
- Converted to:
  - Force in Newtons
  - PC-side elapsed time (`pc_elapsed_s`)
- Logged continuously during detected events

---

### Audio RMS (Sound Level)

Audio loudness is computed as **RMS amplitude per window**, expressed in **dBFS (decibels relative to full scale)**.

#### Calculation

For each RMS window:

```
rms = sqrt(mean(audio_window ** 2))
audio_rms_dbfs = 20 * log10(rms + ε)
```

Where:
- Audio samples are normalized to ±1.0
- `0 dBFS` represents digital clipping
- All real signals are negative dBFS values

#### Interpretation

- dBFS is **relative**, not absolute SPL
- Valid for **comparisons between runs** as long as:
  - Same microphone
  - Same gain settings
  - Same geometry and environment
- Absolute acoustic pressure (dB SPL) requires calibration

---

### Time–Frequency Analysis (Spectrogram)

The spectrogram is computed using a **Short-Time Fourier Transform (STFT)**.

#### Processing Steps

1. Event-only WAV is segmented into overlapping windows
2. Each window is Hann-windowed
3. FFT is computed per window
4. Power spectrum is converted to dB:
   ```
   Power_dB = 10 * log10(|FFT|²)
   ```
5. Frequency bins are mapped vs time

#### What It Shows

- Blade-pass frequency and harmonics
- Broadband noise content
- Evolution of acoustic energy with thrust
- Tonal vs broadband noise characteristics

Because the same processing is applied to all runs, **spectral shape comparisons are valid across propellers**.

---

## GUI Usage

1. Connect STM32 and USB microphone
2. Launch the script:
   ```bash
   python main.py
   ```
3. Enter a run name (e.g., `baseline_5ft_propA`)
4. Click **Start Run**
5. Apply thrust
6. System automatically stops after event ends
7. Outputs are saved to `/runs`

---

## Configuration Parameters

All acquisition parameters are defined at the top of the script:

```python
FORCE_THRESHOLD_N
PRE_SAMPLES
POST_SAMPLES
AUTO_STOP_SECONDS
AUDIO_FS
RMS_WINDOW_S
SPEC_WIN_S
SPEC_OVERLAP
SPEC_NFFT
```

These allow tuning for different test setups without modifying core logic.

---

## Interpretation Guidelines

### What You Can Claim

- Relative loudness differences between propellers
- Frequency content differences
- Tonal vs broadband noise behavior
- Noise evolution with thrust

### What You Cannot Claim (Without Calibration)

- Absolute sound pressure level (dB SPL)
- Compliance with regulatory noise limits
- Direct comparison to manufacturer SPL specs

---

## Dependencies

- Python 3.10+
- numpy
- scipy
- pandas
- matplotlib
- sounddevice
- soundfile
- pyserial
- tkinter (standard library)

Install via:

```bash
pip install numpy scipy pandas matplotlib sounddevice soundfile pyserial
```

---

## Known Limitations

- dBFS is not calibrated SPL
- Spectrogram time base is event-relative, not absolute run time
- Microphone placement must remain fixed between runs
- High-frequency content limited by mic and ADC response

---

## Intended Use Cases

- Propeller comparison studies
- Thrust–acoustic correlation analysis
- Capstone / research test stands
- Rapid prototyping of aeroacoustic experiments

---

## Future Extensions

- SPL calibration support
- Blade-pass frequency overlays using RPM
- Band-integrated noise metrics
- Automated propeller comparison reports

---

## Author Notes

This system was designed to prioritize **repeatability, traceability, and physical relevance**, rather than raw data volume.  
Event-triggered logging ensures that all stored data corresponds directly to meaningful mechanical operation.

