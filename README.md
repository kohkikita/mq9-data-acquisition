
# STM32 Load Cell + Continuous Audio + VESC Control Logger

## Overview

This project implements a multi-modal data acquisition system for propeller and thrust stand testing.

It synchronously captures:

- **Force (N)** from an STM32 load cell system over USB serial
- **Audio** from a USB microphone
- **VESC telemetry** (RPM, voltage, current, duty, MOSFET temperature, input power)
- **Optional VESC motor control** with smooth ramping (Duty / RPM / Current modes)

All signals are time-aligned and post-processed automatically.

Unlike the earlier version of the software, this system is now **continuous-logging**, not event-triggered. Logging starts immediately when **Start Run** is pressed.

---

## System Architecture

```
STM32 Load Cell  ── USB Serial ──┐
                                 ├── Python GUI + DAQ Worker
USB Microphone ── Audio Stream ──┤
                                 └── VESC UART (optional)
                                            │
                                            ▼
                              Continuous Logging + Live Telemetry
                                            │
                                            ▼
                             RPM Plateau Detection / Manual Stop
                                            │
                                            ▼
                              Post-Processing + Auto Outputs
                                            │
                                            ▼
                             CSV + WAV + Spectrogram + Overlays
```

If VESC is enabled:

```
Python → VESC UART (Command Streaming + Telemetry Polling)
```

---

## Major Features

### Continuous Logging

The application now logs continuously from the moment **Start Run** is pressed.

Behavior:

- Logging begins immediately when the run starts
- Force, audio, and VESC telemetry are recorded together during the full run
- The run stops when:
  - the user presses **Stop Run**, or
  - the software detects that **VESC RPM has plateaued**

This matches the newer workflow where the ESC is controlled directly inside the DAQ app.

---

### Audio Capture

- A temporary **FULL WAV** is recorded during the run
- After run completion:
  - The recorded run audio is saved as the final WAV output
  - Spectrogram and audio RMS are computed automatically
  - Temporary files are deleted automatically

---

### VESC Telemetry (Optional)

If enabled, the system logs:

- `vesc_rpm`
- `vesc_v_in_V`
- `vesc_i_motor_A`
- `vesc_i_in_A`
- `vesc_duty`
- `vesc_temp_mos_C`
- `vesc_power_W`

Telemetry is decoded using native VESC packet framing with CRC validation.

---

### VESC Command Modes

Supported modes:

- `disabled`
- `rpm`
- `current`
- `duty`

Duty mode includes:

- Smooth ramp up/down
- Optional hold at final duty
- Optional timed hold
- Sensorless startup assist ("kick")
- Minimum startup duty
- Automatic disarm

Ramp rates and behavior are configurable via GUI.

---

### RPM Plateau Auto-Stop

The system can automatically stop the run when motor RPM appears to have reached its practical limit.

Behavior:

- RPM monitoring arms only after RPM exceeds a minimum threshold
- The software tracks the highest RPM reached so far
- If RPM does not meaningfully exceed that peak for a configured time window, the run automatically stops

Typical parameters:

- `RPM_PLATEAU_MIN_RPM`
- `RPM_PLATEAU_EPS_RPM`
- `RPM_PLATEAU_HOLD_S`
- `RPM_PLATEAU_MIN_DUTY`

---

## Post-Processing Outputs

After each run, the system automatically generates:

### Primary Data Files

| File | Description |
|---|---|
| `*_combined_event_aligned.csv` | Force + audio RMS + interpolated VESC values |
| `*.wav` | Run audio |
| `*_audio_spectrogram.csv` | Spectrogram data |
| `*_audio_spectrogram.png` | Spectrogram plot |

### Overlay Plots

Automatically generated:

- Force + Audio + RPM + Power + Duty
- Force + Audio + RPM
- Force + Audio + Power

---

## Output Directory Structure

```
runs/
├── runname_TIMESTAMP.wav
├── runname_TIMESTAMP_combined_event_aligned.csv
├── runname_TIMESTAMP_audio_spectrogram.csv
├── runname_TIMESTAMP_audio_spectrogram.png
├── runname_TIMESTAMP_overlay_force_audio_rpm_power_duty.png
├── runname_TIMESTAMP_overlay_force_audio_rpm.png
└── runname_TIMESTAMP_overlay_force_audio_power.png
```

Temporary files removed automatically:

- `_TEMP_RAW.csv`
- `_FULL.wav`

---

## GUI Usage

1. Connect STM32
2. Connect USB microphone
3. (Optional) Connect VESC
4. Run:

```
python main.py
```

5. Enter a run name
6. Configure VESC settings if needed
7. Press **Start Run**
8. Apply throttle or allow ramp
9. Run stops when:
   - you press **Stop Run**, or
   - RPM plateau detection triggers

Outputs are saved in `/runs`.

---

## Configuration

Main parameters are located in:

```
daq_app/config.py
```

Examples:

```
BAUD
SERIAL_TIMEOUT_S
FORCE_SCALE
AUDIO_FS
RMS_WINDOW_S
VESC_POLL_HZ
VESC_CMD_HZ
RPM_PLATEAU_MIN_RPM
RPM_PLATEAU_HOLD_S
```

---

## Dependencies

Python 3.10+

Required packages:

```
numpy
scipy
pandas
matplotlib
sounddevice
soundfile
pyserial
tkinter
```

Install:

```
pip install numpy scipy pandas matplotlib sounddevice soundfile pyserial
```

---

## Known Limitations

- dBFS is not calibrated SPL
- Microphone placement must remain fixed between runs
- Spectrogram frequency range limited by microphone response
- RPM plateau detection is heuristic
- If VESC telemetry is unavailable, RPM auto-stop will not trigger

---

## Author Notes

This system was designed for **repeatable propeller testing with synchronized mechanical and acoustic data**.

Continuous logging combined with RPM plateau detection allows clean automated runs while keeping the workflow simple.
