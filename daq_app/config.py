# daq_app/config.py

# ---------------- USER SETTINGS / TUNABLE CONSTANTS ----------------
BAUD = 115200
SERIAL_TIMEOUT_S = 1.0

FORCE_SCALE = 1.143

AUDIO_FS = 48000
AUDIO_CHANNELS = 1
RMS_WINDOW_S = 0.10
EPS = 1e-12

RUNS_DIR = "runs"
PLOT_DPI = 150

SPEC_WIN_S = 0.050
SPEC_OVERLAP = 0.75
SPEC_NFFT = 4096
SPEC_MAX_HZ = None
SPEC_CMAP = "magma"

EVENT_AUDIO_PAD_S = 0.05
MERGE_GAP_S = 0.02

# ---------------- VESC SETTINGS ----------------
# Defaults:
VESC_DEFAULT_ENABLED = True
VESC_DEFAULT_BAUD = 115200
VESC_MODES = ("disabled", "rpm", "current", "duty")

# Background loop rates:
VESC_POLL_HZ = 20.0
VESC_CMD_HZ = 50.0  # higher helps duty ramp feel smoother

# Default GUI values:
VESC_DEFAULT_MODE = "duty"
VESC_DEFAULT_SETPOINT = 1.0
VESC_DEFAULT_RAMP_DUTY_PER_S = 0.05
VESC_DEFAULT_RAMP_RPM_PER_S = 3000.0
VESC_DEFAULT_HOLD_TIME = 3.0
VESC_DEFAULT_RAMP_ENABLE = True
VESC_DEFAULT_HOLD_FINAL = False

# ---------------- RPM PLATEAU AUTO-STOP ----------------
# Run stops automatically if RPM has effectively stopped increasing
# for RPM_PLATEAU_HOLD_S seconds after the monitor is armed.
RPM_PLATEAU_AUTOSTOP_ENABLE = True
RPM_PLATEAU_MIN_RPM = 1000.0        # do not arm below this RPM
RPM_PLATEAU_EPS_RPM = 150.0         # require at least this increase to count as a new peak
RPM_PLATEAU_HOLD_S = 5.0            # auto-stop if no meaningful new peak for this long
RPM_PLATEAU_MIN_DUTY = 0.08         # only evaluate when |duty| is at least this value
RPM_PLATEAU_REQUIRE_VESC = True     # if True, plateau logic only runs when VESC is connected
RPM_PLATEAU_SMOOTH_SAMPLES = 8      # moving average length for RPM smoothing