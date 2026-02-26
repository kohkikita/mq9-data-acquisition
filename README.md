# STM32 Load Cell + Event-Triggered Audio + VESC Control Logger

## Overview

This project implements a multi-modal, event-triggered data acquisition
system for propeller and thrust stand testing.

It synchronously captures:

-   **Force (N)** from an STM32 load cell system over USB serial\
-   **Audio** from a USB microphone (continuous temporary recording →
    trimmed to event-only)\
-   **VESC telemetry** (RPM, voltage, current, duty, MOSFET temperature,
    input power)\
-   **Optional VESC motor control** with smooth ramping (Duty / RPM /
    Current modes)

All signals are time-aligned and post-processed automatically.

------------------------------------------------------------------------

## System Architecture

    STM32 Load Cell  ── USB Serial ──┐
                                     ├── Python GUI + DAQ Worker
    USB Microphone ── Audio Stream ──┘
                                             │
                                             ▼
                                  Event Detection (Force)
                                             │
                                             ▼
                             Post-Processing + Auto Outputs
                                             │
                                             ▼
                             CSV + WAV + Spectrogram + Overlays

If VESC is enabled:

    Python → VESC UART (Command Streaming + Telemetry Polling)

------------------------------------------------------------------------

## Major Features

### Event-Triggered Force Logging

Force events are detected using configurable parameters in `config.py`,
including:

-   `FORCE_THRESHOLD_N`
-   `START_ABOVE_CYCLES`
-   `END_BELOW_CYCLES`
-   `PRE_SAMPLES`
-   `POST_SAMPLES`
-   `AUTO_STOP_SECONDS`

Behavior:

-   Logging begins only after force exceeds threshold for a configured
    number of cycles
-   Includes pre-trigger and post-trigger samples
-   Automatically stops once force falls below threshold for a defined
    duration

------------------------------------------------------------------------

### Audio Capture (Event-Only Saved)

-   A temporary **FULL WAV** is recorded during the run
-   After run completion:
    -   Only event intervals are extracted
    -   An event-only WAV is written
    -   Temporary files are deleted automatically

This reduces disk usage and focuses analysis on physically meaningful
regions.

------------------------------------------------------------------------

### VESC Telemetry (Optional)

If enabled, the system logs:

-   `vesc_rpm`
-   `vesc_v_in_V`
-   `vesc_i_motor_A`
-   `vesc_i_in_A`
-   `vesc_duty`
-   `vesc_temp_mos_C`
-   `vesc_power_W`

Telemetry is decoded using native VESC packet framing with CRC
validation.

------------------------------------------------------------------------

### VESC Command Modes

Supported modes:

-   `disabled`
-   `rpm`
-   `current`
-   `duty`

Duty mode includes:

-   Smooth ramp up/down
-   Optional hold at final duty
-   Optional timed hold
-   Sensorless startup assist ("kick")
-   Minimum startup duty
-   Automatic disarm

Ramp rates and behavior are configurable via GUI.

------------------------------------------------------------------------

## Post-Processing Outputs

After each run, the system automatically generates:

### Primary Data Files

  --------------------------------------------------------------------------------
  File                             Description
  -------------------------------- -----------------------------------------------
  `*_combined_event_aligned.csv`   Force + audio RMS + interpolated VESC values

  `*.wav`                          Event-only audio

  `*_audio_spectrogram.csv`        Spectrogram data

  `*_audio_spectrogram.png`        Spectrogram plot
  --------------------------------------------------------------------------------

### Overlay Plots

Automatically generated:

-   Force + Audio + RPM + Power + Duty
-   Force + Audio + RPM
-   Force + Audio + Power

These provide rapid visual comparison between mechanical and acoustic
behavior.

------------------------------------------------------------------------

## Output Directory Structure

    runs/
    ├── runname_TIMESTAMP.wav
    ├── runname_TIMESTAMP_combined_event_aligned.csv
    ├── runname_TIMESTAMP_audio_spectrogram.csv
    ├── runname_TIMESTAMP_audio_spectrogram.png
    ├── runname_TIMESTAMP_overlay_force_audio_rpm_power_duty.png
    ├── runname_TIMESTAMP_overlay_force_audio_rpm.png
    └── runname_TIMESTAMP_overlay_force_audio_power.png

Run names are user-defined and automatically suffixed with date and time.

Temporary files are deleted automatically:

-   `_TEMP_RAW.csv`
-   `_FULL.wav`


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

1.  Connect STM32
2.  Connect USB microphone
3.  (Optional) Connect VESC
4.  Run:

``` bash
python main.py
```

5.  Enter a run name
6.  Configure VESC (optional)
7.  Click **Start Run**
8.  Apply thrust
9.  System auto-stops after event

Outputs are saved in the `/runs` directory.

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

Python 3.10+

Required packages:

    numpy
    scipy
    pandas
    matplotlib
    sounddevice
    soundfile
    pyserial
    tkinter (standard library)

Install:

``` bash
pip install numpy scipy pandas matplotlib sounddevice soundfile pyserial
```

---

## Known Limitations

- dBFS is not calibrated SPL
- Spectrogram time base is event-relative, not absolute run time
- Microphone placement must remain fixed between runs
- High-frequency content limited by mic and ADC response

---

## Author Notes

This system was designed to prioritize **repeatability, traceability, and physical relevance**, rather than raw data volume.  
Event-triggered logging ensures that all stored data corresponds directly to meaningful mechanical operation.

