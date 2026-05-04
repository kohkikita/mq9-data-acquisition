# STM32 Load Cell + USB Mic + VESC Logger

## Overview

This project is a Python data acquisition application for propeller and thrust stand testing. It records synchronized force, audio, and optional VESC telemetry from the moment a run starts until the run is stopped.

The current workflow is continuous logging. Pressing **Start Run** begins saving data immediately, and the run ends when the user presses **Stop Run** or when RPM plateau auto-stop triggers.

The application captures:

- Force data from an STM32 load cell system over USB serial
- Audio from a USB microphone
- Optional VESC telemetry: RPM, voltage, current, duty, MOSFET temperature, and input power
- Optional VESC motor control in `disabled`, `rpm`, `current`, or `duty` mode

After each run, the app automatically aligns the data streams and generates CSV, WAV, spectrogram, and overlay plot outputs.

## System Architecture

```text
STM32 Load Cell -- USB Serial --+
                                |
USB Microphone -- Audio Stream -+--> Python GUI + DAQ Worker
                                |
VESC UART ------ Optional ------+

Python GUI + DAQ Worker
        |
        +--> Continuous logging
        +--> Optional VESC command ramp
        +--> Manual stop or RPM plateau auto-stop
        +--> Post-processing
        +--> CSV + WAV + spectrogram + overlay plots
```

## Major Features

### Continuous Logging

Logging starts as soon as **Start Run** is pressed. All valid force samples, audio, and available VESC telemetry are saved for the full run.

The run stops when:

- The user presses **Stop Run**
- RPM plateau detection decides the motor speed has stopped meaningfully increasing

Internally, the saved run is still represented with `event_id = 1` so the existing post-processing code can reuse the older event-alignment path. Some output filenames still include `event_aligned`, but they now describe the full saved run rather than a force-triggered event.

### GUI

The GUI provides:

- Dark red/black theme
- Start/stop controls and run name entry
- Status indicator with state color
- Ramp progress bar for VESC ramping
- RPM plateau auto-stop checkbox
- Responsive VESC and DAQ settings panels
- Live RPM and power plot with project logo watermark
- Compact live log capped to about six visible lines

### Audio Capture

A temporary full-run WAV is recorded while the run is active. During post-processing, the final run WAV is written, audio RMS is calculated, and spectrogram outputs are generated.

Temporary audio and raw CSV files are removed automatically after post-processing completes.

### VESC Telemetry and Control

When VESC support is enabled, the system can poll telemetry and stream motor commands at configurable rates.

Logged telemetry fields include:

- `vesc_rpm`
- `vesc_v_in_V`
- `vesc_i_motor_A`
- `vesc_i_in_A`
- `vesc_duty`
- `vesc_temp_mos_C`
- `vesc_power_W`

Supported command modes:

- `disabled`
- `rpm`
- `current`
- `duty`

Duty mode supports smooth ramping, optional final-duty hold, sensorless startup assist, minimum startup duty, and automatic disarm behavior.

### RPM Plateau Auto-Stop

RPM plateau auto-stop is used to end a run automatically when RPM stops increasing meaningfully.

The logic:

- Waits until smoothed RPM exceeds `RPM_PLATEAU_MIN_RPM`
- Requires duty to exceed `RPM_PLATEAU_MIN_DUTY`
- Tracks the highest RPM reached so far
- Stops the run if RPM does not exceed the previous peak by `RPM_PLATEAU_EPS_RPM` for `RPM_PLATEAU_HOLD_S`

If VESC telemetry is unavailable and `RPM_PLATEAU_REQUIRE_VESC` is enabled, plateau auto-stop will not trigger.

## Outputs

Each run writes files to `runs/` using the run name and timestamp.

```text
runs/
|-- runname_TIMESTAMP.wav
|-- runname_TIMESTAMP_combined_event_aligned.csv
|-- runname_TIMESTAMP_audio_spectrogram.csv
|-- runname_TIMESTAMP_audio_spectrogram.png
|-- runname_TIMESTAMP_overlay_force_audio_rpm_power_duty.png
|-- runname_TIMESTAMP_overlay_force_audio_rpm.png
`-- runname_TIMESTAMP_overlay_force_audio_power.png
```

Primary outputs:

| File | Description |
|---|---|
| `*.wav` | Final run audio |
| `*_combined_event_aligned.csv` | Full-run force, audio RMS, and interpolated VESC data. The filename is legacy. |
| `*_audio_spectrogram.csv` | Spectrogram data |
| `*_audio_spectrogram.png` | Spectrogram plot |
| `*_overlay_force_audio_rpm_power_duty.png` | Force, audio RMS, RPM, power, and duty overlay |
| `*_overlay_force_audio_rpm.png` | Force, audio RMS, and RPM overlay |
| `*_overlay_force_audio_power.png` | Force, audio RMS, and power overlay |

Temporary files removed after post-processing:

- `_TEMP_RAW.csv`
- `_FULL.wav`

## GUI Usage

1. Connect the STM32 load cell system.
2. Connect the USB microphone.
3. Optionally connect the VESC.
4. Run the app:

```bash
python main.py
```

5. Enter a run name.
6. Configure VESC settings if needed.
7. Press **Start Run**.
8. Let the run complete, or press **Stop Run**.
9. Review generated outputs in `runs/`.

## Configuration

Main settings are in `daq_app/config.py`.

Common DAQ settings:

```python
BAUD
SERIAL_TIMEOUT_S
FORCE_SCALE
AUDIO_FS
AUDIO_CHANNELS
RMS_WINDOW_S
RUNS_DIR
```

VESC settings:

```python
VESC_DEFAULT_ENABLED
VESC_DEFAULT_BAUD
VESC_MODES
VESC_POLL_HZ
VESC_CMD_HZ
VESC_DEFAULT_MODE
VESC_DEFAULT_SETPOINT
VESC_DEFAULT_RAMP_DUTY_PER_S
VESC_DEFAULT_RAMP_RPM_PER_S
VESC_DEFAULT_HOLD_TIME
VESC_DEFAULT_RAMP_ENABLE
VESC_DEFAULT_HOLD_FINAL
```

RPM plateau settings:

```python
RPM_PLATEAU_AUTOSTOP_ENABLE
RPM_PLATEAU_MIN_RPM
RPM_PLATEAU_EPS_RPM
RPM_PLATEAU_HOLD_S
RPM_PLATEAU_MIN_DUTY
RPM_PLATEAU_REQUIRE_VESC
RPM_PLATEAU_SMOOTH_SAMPLES
```

The STM32 serial parser expects load-cell lines in this general form:

```text
Load=0.123 N, t=4567 ms
```

## Dependencies

Python 3.10+ is recommended.

Install Python package dependencies:

```bash
pip install -r requirements.txt
```

`tkinter` is used for the GUI and ships with most Windows Python installs. On some Linux distributions, it may need to be installed through the system package manager.

## Known Limitations

- Audio dBFS values are not calibrated SPL.
- Microphone placement must remain fixed between runs for meaningful acoustic comparisons.
- Spectrogram frequency range depends on the microphone and sample rate.
- RPM plateau detection is heuristic and depends on valid VESC telemetry.
- The post-processing path still uses legacy `event` names even though the current run data is continuous.

## Author Notes

This system is designed for repeatable propeller testing with synchronized mechanical, acoustic, and motor telemetry data. Continuous logging keeps the test workflow simple, while RPM plateau auto-stop allows automated run completion when VESC telemetry is available.
