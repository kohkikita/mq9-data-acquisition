# Python Script for MQ-9A Propeller Optimization Data Acquisition
**SDSU Engineering | General Atomics**

---

## Requirements

- Python 3.7+
- A connected STM32 microcontroller (e.g. Nucleo board) outputting serial data in the format:
  ```
  Load=0.123 N, t=4567 ms
  ```
- A USB microphone or audio input device
- The following Python packages (installed via `requirements.txt`):

| Package | Purpose |
|---|---|
| `numpy` | Numerical processing |
| `pandas` | CSV handling and data alignment |
| `scipy` | WAV file reading |
| `pyserial` | STM32 serial communication |
| `sounddevice` | Audio input stream |
| `soundfile` | WAV file writing |
| `matplotlib` | Post-run plot generation |
| `tkinter` | GUI (bundled with most Python installs) |

---

## How to Download the Script

1. Create a folder in the desired location on your machine
2. Open the folder in VSCode
3. Open the terminal (`Cmd/Ctrl + J`)
4. Clone the repository:

```bash
git clone https://github.com/kohkikita/mq9-data-acquisition.git
```

5. Verify you are in the correct directory — your terminal should show: `mq9-data-acquisition`
   - If not, use `cd` to navigate into the correct directory (press `Tab` to autocomplete directory names)
6. Once in the correct directory, install the required dependencies:

```bash
pip3 install -r requirements.txt
```

---

## How to Run the Script

```bash
python3 serialLogger.py
```

A GUI window will open. Press **Start Run** to begin recording.

---

## Configuration

All tunable parameters are located at the top of `serialLogger.py` under `USER SETTINGS`. No other part of the file needs to be edited for normal use.

| Parameter | Default | Description |
|---|---|---|
| `BAUD` | `115200` | Serial baud rate — must match STM32 firmware |
| `FORCE_THRESHOLD_N` | `0.4` | Force level (N) that triggers an event |
| `PRE_SAMPLES` | `25` | Samples captured before event trigger |
| `POST_SAMPLES` | `25` | Samples captured after event ends |
| `AUTO_STOP_SECONDS` | `1.0` | Seconds below threshold before auto-stopping |
| `START_ABOVE_CYCLES` | `15` | Consecutive samples above threshold required to start an event |
| `END_BELOW_CYCLES` | `15` | Consecutive samples below threshold required to end an event |
| `AUDIO_FS` | `48000` | Audio sample rate (Hz) |
| `RMS_WINDOW_S` | `0.10` | RMS window size for audio post-processing (seconds) |
| `RUNS_DIR` | `"runs"` | Output folder for all run files |

---

## Output Files

Each run produces 4 files inside the `runs/` folder, all sharing a timestamped base name (e.g. `loadcell_run_2025-01-01_12-00-00`):

| File | Description |
|---|---|
| `***.csv` | Raw event CSV — force and timing data for all captured events |
| `***.wav` | Full audio recording for the run |
| `***_combined_event_aligned.csv` | Post-processed CSV — force and audio RMS aligned by time window |
| `***_force_audio_plot.png` | Plot of force (N) vs. audio level (dBFS) over the event |

---

## Common Issues

**No serial ports found** — Make sure the STM32 is connected via USB and powered on before starting the script.

**Wrong serial port selected** — The script auto-detects STM32/Nucleo/ST-Link descriptors. If detection fails and only one port exists, it will use that port. If multiple non-STM32 ports exist, the script will raise an error listing available ports — manually set the correct port in the code if needed.

**No audio input detected** — The script uses your system default audio input device. If you need a specific microphone, set `mic_device` to the correct device index in the `start_run()` method. You can list available devices by running:
```python
import sounddevice as sd
print(sd.query_devices())
```

**tkinter not found** — On Linux, tkinter may need to be installed separately:
```bash
sudo apt-get install python3-tk
```

---

## ⚠️ Warnings

- This README uses `pip3` and `python3`. If you do not have `pip3` or `python3` installed, replace all instances with `pip` and `python` respectively.
- The STM32 firmware **must** output data in the exact format `Load=X.XXX N, t=XXXX ms` or the script will not parse any values.
- Do **not** close the GUI window mid-run — use the **Stop Run** button to ensure audio and serial connections are properly closed and post-processing completes.
```