# daq_app/config.py
import re

# ---------------- USER SETTINGS / TUNABLE CONSTANTS ----------------
BAUD = 115200
SERIAL_TIMEOUT_S = 1.0

# Trigger / capture behavior (event-only logging)
FORCE_THRESHOLD_N = 0.8
PRE_SAMPLES = 25
POST_SAMPLES = 25
AUTO_STOP_SECONDS = 1.0

START_ABOVE_CYCLES = 15
END_BELOW_CYCLES = 15

# Audio / post-processing
AUDIO_FS = 48000
AUDIO_CHANNELS = 1
RMS_WINDOW_S = 0.10
EPS = 1e-12

# Output directory
RUNS_DIR = "runs"

# Plot settings
PLOT_DPI = 150

# Spectrogram (time-frequency) settings
SPEC_WIN_S = 0.050
SPEC_OVERLAP = 0.75
SPEC_NFFT = 4096
SPEC_MAX_HZ = None
SPEC_CMAP = "magma"

# Audio trimming behavior (event-only WAV)
EVENT_AUDIO_PAD_S = 0.05
MERGE_GAP_S = 0.02

# ---------------- VESC SETTINGS ----------------
VESC_DEFAULT_ENABLED = False
VESC_DEFAULT_BAUD = 115200
VESC_POLL_HZ = 20.0
VESC_CMD_HZ = 20.0

# Supported control modes in this script
VESC_MODES = ("disabled", "rpm", "current", "duty")

# ---------------- VESC RAMP / HOLD SETTINGS ----------------
VESC_RAMP_ENABLE = True
VESC_RAMP_RPM_PER_S = 3000.0
VESC_RAMP_DUTY_PER_S = 0.10

VESC_MAX_RPM = 40000
VESC_MAX_DUTY = 0.60

VESC_HOLD_FINAL_DUTY = True

# STM32 parsing
LINE_RE = re.compile(r"Load=([+-]?\d+(?:\.\d+)?)\s*N,\s*t=(\d+)\s*ms")
