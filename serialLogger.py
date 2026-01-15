"""
STM32 Load Cell + USB Microphone Data Acquisition

High-level pipeline:

  1. GUI launches a background RunWorker thread.
  2. RunWorker:
       - Opens STM32 serial port and starts a FULL audio recording.
       - Runs a small state machine on incoming force samples to detect "events":
            * Forces above FORCE_THRESHOLD_N for START_ABOVE_CYCLES samples start an event.
            * Forces below FORCE_THRESHOLD_N for END_BELOW_CYCLES samples start the event tail.
            * POST_SAMPLES additional samples are recorded after the tail.
       - Writes all in-event samples (plus pre-buffered samples) to a raw CSV.
       - After all events (or stop), stops audio + serial.
       - From FULL WAV + raw CSV:
            * Builds a time-aligned combined CSV (force + audio RMS vs time).
            * Saves a force/audio plot.
            * Extracts only the audio that overlaps events into a final event-only WAV.
            * Computes an STFT spectrogram from the event-only WAV.
       - Deletes the FULL WAV to save disk space.

Result per run:
  - Raw force CSV.
  - Event-only WAV.
  - Combined force/audio CSV (+ dual-axis PNG).
  - Spectrogram CSV (+ PNG) for frequency-vs-time analysis.
"""

import os
import re
import time
import csv
import queue
import threading
import math
from dataclasses import dataclass
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from scipy.io import wavfile

import serial
from serial.tools import list_ports

import sounddevice as sd
import soundfile as sf

import tkinter as tk
from tkinter import ttk, messagebox

# Plotting (headless-safe for Tkinter apps on Windows / remote)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS / TUNABLE CONSTANTS ----------------
BAUD = 115200
SERIAL_TIMEOUT_S = 1.0

# Trigger / capture behavior (event-only logging)
# Forces above FORCE_THRESHOLD_N are treated as "interesting" (prop under load).
FORCE_THRESHOLD_N = 0.4
PRE_SAMPLES = 25          # number of samples buffered before event start
POST_SAMPLES = 25         # number of samples to keep after event end
AUTO_STOP_SECONDS = 1.0   # after any event: stop run if below threshold for this long

# Debounced start/end conditions
# These prevent single noisy samples from starting/ending an event.
START_ABOVE_CYCLES = 15   # require N consecutive samples >= threshold to start event
END_BELOW_CYCLES = 15     # require N consecutive samples < threshold to end event

# Audio / post-processing
AUDIO_FS = 48000          # USB microphone sample rate
AUDIO_CHANNELS = 1
RMS_WINDOW_S = 0.10       # window size for RMS / dBFS calculation
EPS = 1e-12               # small constant to avoid log10(0)

# Output directory
RUNS_DIR = "runs"

# Plot settings
PLOT_DPI = 150

# Spectrogram (time-frequency) settings
SPEC_WIN_S = 0.050        # STFT window length in seconds (e.g., 50 ms)
SPEC_OVERLAP = 0.75       # fraction overlap between consecutive windows (0–<1)
SPEC_NFFT = 4096          # FFT size (>= window length in samples)
SPEC_MAX_HZ = None        # e.g., 10000 to limit spectrogram to 10 kHz; None = Nyquist
SPEC_CMAP = "magma"

# Audio trimming behavior (event-only WAV)
EVENT_AUDIO_PAD_S = 0.05  # pad each event interval on both sides when extracting audio
MERGE_GAP_S = 0.02        # merge event intervals if gap between them <= this (seconds)
# -------------------------------------------------------------------

# Matches STM32 lines like: "Load=0.123 N, t=4567 ms"
LINE_RE = re.compile(r"Load=([+-]?\d+(?:\.\d+)?)\s*N,\s*t=(\d+)\s*ms")


def ensure_dir(path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    """Return current date/time as filesystem-friendly string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sanitize_run_name(name: str) -> str:
    """
    Make a filesystem-safe run name:
      - trims whitespace
      - spaces -> underscores
      - keeps only letters/numbers/_/-
      - falls back to "run" if empty

    This is used as the human-chosen prefix before the timestamp.
    """
    name = (name or "").strip()
    if not name:
        return "run"
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "run"


def find_stm32_port() -> str:
    """
    Try to automatically locate the STM32 serial port.

    Strategy:
      1. Prefer ports whose description/manufacturer mention STM / Nucleo / STLink.
      2. If only one port exists, use it.
      3. Otherwise raise with a helpful list of candidates.
    """
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found.")

    # Prefer ST/STM32 descriptors
    for p in ports:
        desc = (p.description or "").lower()
        manu = (p.manufacturer or "").lower()
        if any(k in desc for k in ["stm", "stlink", "nucleo", "stm32"]) or \
           any(k in manu for k in ["stmicroelectronics", "st"]):
            return p.device

    # If only one port exists, use it
    if len(ports) == 1:
        return ports[0].device

    # Otherwise ambiguous: show all possibilities
    lines = ["Could not uniquely identify STM32 serial port.", "Available ports:"]
    for p in ports:
        lines.append(f"  {p.device}: {p.description} ({p.manufacturer})")
    raise RuntimeError("\n".join(lines))


def parse_line(line: str):
    """
    Parse a line of STM32 output of the form:
        'Load=<float> N, t=<int> ms'
    Returns:
        (force_N: float, t_ms: int) or None if the line does not match.
    """
    m = LINE_RE.search(line)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


@dataclass
class RunPaths:
    """
    Container for all file paths associated with a single run.
    """
    base_name: str
    raw_event_csv: str
    wav_full_path: str      # temporary: full recording during run (deleted at end)
    wav_event_path: str     # final: ONLY event audio (what you keep)
    combined_csv: str
    plot_png: str
    spectrogram_csv: str
    spectrogram_png: str


def make_run_paths(run_name: str) -> RunPaths:
    """
    Construct all output paths for a new run.

    Naming convention:
      <run_name>_<YYYY-MM-DD>_<HH-MM-SS>.*

    The timestamp always appears at the end of the base prefix so runs can
    be sorted lexicographically by time.
    """
    ensure_dir(RUNS_DIR)
    stamp = now_stamp()
    run_name = sanitize_run_name(run_name)
    base = f"{run_name}_{stamp}"

    raw_csv = os.path.join(RUNS_DIR, f"{base}.csv")

    # Full audio is recorded to a temp file, then trimmed to event-only wav_event_path.
    wav_full = os.path.join(RUNS_DIR, f"{base}_FULL.wav")
    wav_event = os.path.join(RUNS_DIR, f"{base}.wav")

    combined = os.path.join(RUNS_DIR, f"{base}_combined_event_aligned.csv")
    plot_png = os.path.join(RUNS_DIR, f"{base}_force_audio_plot.png")

    spec_csv = os.path.join(RUNS_DIR, f"{base}_audio_spectrogram.csv")
    spec_png = os.path.join(RUNS_DIR, f"{base}_audio_spectrogram.png")

    return RunPaths(base, raw_csv, wav_full, wav_event, combined, plot_png, spec_csv, spec_png)


def postprocess_event_aligned(
    force_csv_path: str,
    wav_path: str,
    out_csv_path: str,
    rms_window_s: float = RMS_WINDOW_S,
) -> None:
    """
    Build an event-aligned combined CSV where each row corresponds to
    a fixed audio RMS window and the force has been interpolated to the
    center of that window.

    - Uses 'pc_elapsed_s' from the force CSV as the common timebase.
    - Interprets wav_path as starting near the run start (pc_elapsed_s ~ 0).
      Therefore we run this on the FULL WAV, not the trimmed event-only WAV.

    Output columns:
      - t_event_s        : time (s) relative to first included window
      - force_N          : interpolated force at the window center
      - audio_rms_dbfs   : windowed RMS in dBFS (relative to full scale)
      - audio_rms_linear : linear RMS amplitude
      - window_start_s   : window start (relative to t_event_s=0)
      - window_end_s     : window end   (relative to t_event_s=0)
    """
    df_force = pd.read_csv(force_csv_path)
    if "pc_elapsed_s" not in df_force.columns or "force_N" not in df_force.columns:
        raise RuntimeError("Force CSV missing required columns: pc_elapsed_s, force_N")

    df_force["pc_elapsed_s"] = df_force["pc_elapsed_s"].astype(float)
    df_force["force_N"] = df_force["force_N"].astype(float)

    t_force_min = float(df_force["pc_elapsed_s"].min())
    t_force_max = float(df_force["pc_elapsed_s"].max())

    fs, audio = wavfile.read(wav_path)

    # Convert audio to float [-1,1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]

    win_len = int(rms_window_s * fs)
    if win_len <= 0:
        raise RuntimeError("Invalid RMS window length.")

    n_windows = len(audio) // win_len
    if n_windows <= 0:
        raise RuntimeError("Audio too short for chosen RMS window.")

    # Determine the first/last audio window whose midpoint lies inside
    # the span of the force measurements.
    first_i = None
    last_i = None
    for i in range(n_windows):
        t_mid = (i * rms_window_s) + 0.5 * rms_window_s
        if t_force_min <= t_mid <= t_force_max:
            first_i = i
            break
    for i in range(n_windows - 1, -1, -1):
        t_mid = (i * rms_window_s) + 0.5 * rms_window_s
        if t_force_min <= t_mid <= t_force_max:
            last_i = i
            break

    if first_i is None or last_i is None:
        raise RuntimeError("No audio windows overlap the force event window.")

    # Define t=0 for the combined dataset as the start of the first included window.
    t0_event = first_i * rms_window_s

    force_t = df_force["pc_elapsed_s"].values
    force_y = df_force["force_N"].values

    rows = []
    for i in range(first_i, last_i + 1):
        t_start = i * rms_window_s
        t_end = (i + 1) * rms_window_s
        t_mid = t_start + 0.5 * rms_window_s

        seg = audio[i * win_len:(i + 1) * win_len]
        rms_lin = float(np.sqrt(np.mean(seg ** 2)))
        rms_dbfs = float(20 * np.log10(rms_lin + EPS))

        # Interpolate the force to the audio window midpoint.
        force_N = float(np.interp(t_mid, force_t, force_y))

        rows.append({
            "t_event_s": t_mid - t0_event,
            "force_N": force_N,
            "audio_rms_dbfs": rms_dbfs,
            "audio_rms_linear": rms_lin,
            "window_start_s": t_start - t0_event,
            "window_end_s": t_end - t0_event,
        })

    df = pd.DataFrame(rows)

    # Round to manageable precision for downstream analysis.
    df["t_event_s"] = df["t_event_s"].round(3)
    df["window_start_s"] = df["window_start_s"].round(3)
    df["window_end_s"] = df["window_end_s"].round(3)
    df["force_N"] = df["force_N"].round(3)
    df["audio_rms_dbfs"] = df["audio_rms_dbfs"].round(2)
    df["audio_rms_linear"] = df["audio_rms_linear"].round(6)

    df.to_csv(out_csv_path, index=False)


def save_force_audio_plot(
    combined_csv_path: str,
    out_png_path: str,
    title: str = "Force vs Audio RMS (Post-Processed)",
) -> None:
    """
    Read the combined CSV and create a dual-axis plot:
      - Audio RMS (dBFS) vs time (orange, left axis)
      - Force (N) vs time (blue, right axis)

    This is the main visualization for "loudness vs thrust" behavior.
    """
    df = pd.read_csv(combined_csv_path)

    required = {"t_event_s", "force_N", "audio_rms_dbfs"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Combined CSV missing required columns: {sorted(required)}")

    t = df["t_event_s"].astype(float).values
    force = df["force_N"].astype(float).values
    audio_db = df["audio_rms_dbfs"].astype(float).values

    fig, ax_audio = plt.subplots(figsize=(10, 5.2))
    ax_force = ax_audio.twinx()

    # Explicit colors + labels for clarity.
    ax_audio.plot(t, audio_db, color="tab:orange", linewidth=1.8, label="Audio (dBFS)")
    ax_force.plot(t, force, color="tab:blue", linewidth=1.8, label="Force (N)")

    ax_audio.set_title(title)
    ax_audio.set_xlabel("Time (s)")
    ax_audio.set_ylabel("Audio Level (dBFS)", color="tab:orange")
    ax_force.set_ylabel("Force (N)", color="tab:blue")

    ax_audio.tick_params(axis="y", colors="tab:orange")
    ax_force.tick_params(axis="y", colors="tab:blue")

    ax_audio.grid(True, which="both", alpha=0.3)

    # Combined legend from both axes.
    h1, l1 = ax_audio.get_legend_handles_labels()
    h2, l2 = ax_force.get_legend_handles_labels()
    ax_audio.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)


def compute_audio_spectrogram(
    wav_path: str,
    out_csv_path: str,
    out_png_path: str,
    *,
    win_s: float = SPEC_WIN_S,
    overlap: float = SPEC_OVERLAP,
    nfft: int = SPEC_NFFT,
    max_hz: float | None = SPEC_MAX_HZ,
) -> None:
    """
    Compute an STFT spectrogram of the given WAV and save:

      - CSV (long format):
          t_s      : time (center of STFT window)
          freq_hz  : frequency bin center
          mag_db   : power magnitude in dB (relative, not SPL)
      - PNG: time-frequency heatmap (spectrogram).

    The input WAV is expected to be the "event-only" WAV generated by
    write_event_only_wav, so its time axis corresponds to "event-audio
    timeline" with gaps removed.
    """
    if not os.path.exists(wav_path):
        raise RuntimeError(f"WAV not found: {wav_path}")

    fs, audio = wavfile.read(wav_path)

    # Mono only (take first channel if multi-channel).
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Normalize to float [-1, 1] for all integer PCM formats.
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float64) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float64)

    # Remove NaNs / infs just in case.
    audio = audio[np.isfinite(audio)]
    if audio.size == 0:
        raise RuntimeError("Audio empty.")

    win_len = int(round(win_s * fs))
    if win_len < 16:
        raise RuntimeError("SPEC_WIN_S too small (window < 16 samples).")

    # Ensure nfft is at least as large as window size.
    if nfft < win_len:
        nfft = int(2 ** math.ceil(math.log2(win_len)))

    hop = int(round(win_len * (1.0 - overlap)))
    hop = max(1, hop)

    window = np.hanning(win_len)

    # Number of STFT frames that fit in the audio buffer.
    n_frames = 1 + (audio.size - win_len) // hop if audio.size >= win_len else 0
    if n_frames <= 0:
        raise RuntimeError("Audio too short for chosen spectrogram window.")

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Optional frequency limit: restrict to <= max_hz.
    if max_hz is not None:
        fmask = freqs <= float(max_hz)
    else:
        fmask = slice(None)

    freqs_use = freqs[fmask]

    # spec_db shape: [frequency_bins, time_frames]
    spec_db = np.empty((freqs_use.size, n_frames), dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        seg = audio[start:start + win_len]
        segw = seg * window

        X = np.fft.rfft(segw, n=nfft)
        P = (np.abs(X) ** 2)  # power spectrum (relative units)

        P = P[fmask]
        spec_db[:, i] = 10.0 * np.log10(np.maximum(P, 1e-20))

    # Time axis in seconds (center of each analysis window).
    t_s = (np.arange(n_frames) * hop + (win_len / 2.0)) / fs

    # ---- Save CSV (long format) ----
    chunks = []
    for ti in range(n_frames):
        chunks.append(
            pd.DataFrame(
                {
                    "t_s": np.full(freqs_use.size, t_s[ti], dtype=np.float64),
                    "freq_hz": freqs_use.astype(np.float64),
                    "mag_db": spec_db[:, ti].astype(np.float64),
                }
            )
        )
    df = pd.concat(chunks, ignore_index=True)
    df["t_s"] = df["t_s"].round(4)
    df["freq_hz"] = df["freq_hz"].round(2)
    df["mag_db"] = df["mag_db"].round(2)
    df.to_csv(out_csv_path, index=False)

    # ---- Save PNG ----
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    im = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=[t_s[0], t_s[-1], freqs_use[0], freqs_use[-1]],
        cmap=SPEC_CMAP,
    )
    ax.set_title("Audio Spectrogram (STFT) - Event Audio Only")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="Magnitude (dB, relative)")
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)


def get_event_intervals_from_raw_csv(raw_event_csv: str) -> list[tuple[float, float]]:
    """
    From the raw event CSV, compute a list of (start, end) intervals in pc_elapsed_s
    that correspond to detected events.

    Steps:
      1. Filter to rows with event_id >= 1.
      2. For each event_id, take [min(pc_elapsed_s), max(pc_elapsed_s)].
      3. Pad each interval by EVENT_AUDIO_PAD_S on both sides.
      4. Merge any intervals that overlap or are separated by <= MERGE_GAP_S.

    Returned list is sorted by start time and is used to trim the full WAV
    down to an event-only WAV.
    """
    df = pd.read_csv(raw_event_csv)
    if df.empty:
        return []

    if "event_id" not in df.columns or "pc_elapsed_s" not in df.columns:
        raise RuntimeError("Raw event CSV missing required columns: event_id, pc_elapsed_s")

    df = df.copy()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)
    df["pc_elapsed_s"] = pd.to_numeric(df["pc_elapsed_s"], errors="coerce").astype(float)

    df = df[df["event_id"] >= 1]
    if df.empty:
        return []

    grp = df.groupby("event_id")["pc_elapsed_s"]
    intervals = []
    for _, s in grp:
        a = float(s.min()) - EVENT_AUDIO_PAD_S
        b = float(s.max()) + EVENT_AUDIO_PAD_S
        intervals.append((max(0.0, a), max(0.0, b)))

    intervals.sort(key=lambda x: x[0])

    # Merge overlapping / close intervals to avoid tiny gaps in the event WAV.
    merged: list[tuple[float, float]] = []
    for a, b in intervals:
        if not merged:
            merged.append((a, b))
            continue
        pa, pb = merged[-1]
        if a <= pb + MERGE_GAP_S:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))

    return merged


def write_event_only_wav(
    full_wav_path: str,
    out_wav_path: str,
    intervals_s: list[tuple[float, float]],
) -> None:
    """
    Extract all samples that fall inside the given time intervals from
    the full WAV and concatenate them into a single "event-only" WAV.

    The resulting WAV's time axis is "event-audio timeline" — i.e. all
    idle periods between events are removed.
    """
    if not os.path.exists(full_wav_path):
        raise RuntimeError(f"Full WAV not found: {full_wav_path}")

    fs, audio = wavfile.read(full_wav_path)

    if audio.ndim > 1:
        audio = audio[:, 0]

    # Ensure we output int16 PCM (consistent with AudioRecorder).
    if not np.issubdtype(audio.dtype, np.integer):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    elif audio.dtype != np.int16:
        maxv = np.iinfo(audio.dtype).max
        audio_f = audio.astype(np.float64) / maxv
        audio = np.clip(audio_f, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)

    n = audio.shape[0]
    if n == 0:
        raise RuntimeError("Full WAV is empty.")

    segs = []
    for a_s, b_s in intervals_s:
        i0 = int(round(a_s * fs))
        i1 = int(round(b_s * fs))
        i0 = max(0, min(n, i0))
        i1 = max(0, min(n, i1))
        if i1 > i0:
            segs.append(audio[i0:i1])

    if not segs:
        raise RuntimeError("No audio samples found within event intervals (nothing to save).")

    out = np.concatenate(segs, axis=0)
    wavfile.write(out_wav_path, fs, out)


class AudioRecorder:
    """
    Helper class that records audio to disk using sounddevice + soundfile.

    Usage:
        rec = AudioRecorder(path, fs, channels)
        rec.start()
        ...
        rec.stop()
    """

    def __init__(self, wav_path: str, fs: int, channels: int, device=None):
        self.wav_path = wav_path
        self.fs = fs
        self.channels = channels
        self.device = device

        self._sf = None
        self._stream = None

    def start(self):
        """Open a soundfile and start a sounddevice InputStream writing into it."""
        self._sf = sf.SoundFile(
            self.wav_path, mode="w", samplerate=self.fs, channels=self.channels, subtype="PCM_16"
        )

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            self._sf.write(indata)

        self._stream = sd.InputStream(
            samplerate=self.fs,
            channels=self.channels,
            device=self.device,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop(self):
        """Stop the audio stream and close the underlying file."""
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            if self._sf is not None:
                self._sf.close()
                self._sf = None


class RunWorker(threading.Thread):
    """
    Background worker that implements one complete run.

    Responsibilities:
      - Open STM32 serial port.
      - Start FULL audio recording (temporary WAV).
      - Run an event-detection state machine on force samples with:
            * Pre-buffering
            * Start/end debounce
            * Post-event tail
            * Auto-stop after being below threshold for AUTO_STOP_SECONDS.
      - Write raw event CSV during run.
      - After the run:
            * Create combined CSV (force + audio RMS) from FULL WAV.
            * Save force/audio plot.
            * Extract event-only WAV (trimmed from FULL WAV).
            * Compute spectrogram from event-only WAV.
            * Delete FULL WAV to keep only event-related artifacts.
    """

    def __init__(self, gui_queue: queue.Queue, paths: RunPaths, mic_device=None):
        super().__init__(daemon=True)
        self.gui_queue = gui_queue
        self.paths = paths
        self.mic_device = mic_device
        self._stop_req = threading.Event()

    def request_stop(self):
        """Signal the worker loop to stop at the next safe opportunity."""
        self._stop_req.set()

    def run(self):
        """
        Main worker loop:

          1. Connect to STM32 and start audio.
          2. For each line from serial, parse force + time and feed into the
             event detection logic (state machine).
          3. When done, stop audio, close serial, and run all post-processing.
        """
        try:
            port = find_stm32_port()
            self.gui_queue.put(("status", f"Serial: {port} @ {BAUD}"))

            ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT_S)
            time.sleep(2.0)  # STM32 often resets on open; allow it to boot

            # Use a single time origin for pc_elapsed_s.
            t0 = time.perf_counter()

            # Start FULL audio (temporary; later trimmed to events).
            self.gui_queue.put(("status", "Starting audio (temporary full recording)..."))
            audio = AudioRecorder(self.paths.wav_full_path, AUDIO_FS, AUDIO_CHANNELS, device=self.mic_device)
            audio.start()

            # Event capture state / debouncing.
            pre_buffer = deque(maxlen=PRE_SAMPLES)  # rolling buffer of pre-event samples
            in_event = False
            post_remaining = 0
            event_id = 0
            saw_any_event = False
            below_since = None  # wall-clock time when we first went below threshold

            # Debounce counters
            above_count = 0  # consecutive samples >= threshold (for event start)
            below_count = 0  # consecutive samples < threshold (for event end)

            with open(self.paths.raw_event_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["event_id", "pc_time_iso", "pc_elapsed_s", "stm32_time_ms", "force_N", "raw_line"])

                self.gui_queue.put(("status", f"Recording... ({self.paths.base_name})"))

                while not self._stop_req.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode(errors="ignore").strip()
                    if not line:
                        continue

                    parsed = parse_line(line)
                    if not parsed:
                        continue

                    force_N, stm32_ms = parsed
                    t_elapsed = time.perf_counter() - t0
                    now_iso = datetime.now().isoformat()

                    # Display a human-friendly status line in the GUI log.
                    self.gui_queue.put(("line", f"t={stm32_ms:8d} ms | pc={t_elapsed:8.3f} s | F={force_N:7.3f} N"))

                    row = [event_id, now_iso, t_elapsed, stm32_ms, force_N, line]
                    now_pc = time.perf_counter()

                    if not in_event:
                        # We are currently outside any event: update pre-buffer and
                        # monitor for event start.
                        pre_buffer.append(row)

                        # Start debounce: require sustained force above threshold.
                        if force_N >= FORCE_THRESHOLD_N:
                            above_count += 1
                            if above_count >= START_ABOVE_CYCLES:
                                # Event officially starts.
                                in_event = True
                                post_remaining = 0
                                event_id += 1
                                saw_any_event = True
                                below_since = None

                                below_count = 0
                                self.gui_queue.put(
                                    (
                                        "status",
                                        f"EVENT {event_id} START (above for {START_ABOVE_CYCLES} cycles)",
                                    )
                                )

                                # Flush pre-buffer into the CSV as part of the new event.
                                for r in pre_buffer:
                                    r[0] = event_id
                                    writer.writerow(r)
                                f.flush()
                                pre_buffer.clear()

                                # Also write the current sample as part of the event.
                                writer.writerow([event_id, now_iso, t_elapsed, stm32_ms, force_N, line])
                                f.flush()
                        else:
                            # Force below threshold: reset start-debounce counter.
                            above_count = 0

                            # Auto-stop: once we've seen at least one event, if we stay
                            # below threshold for AUTO_STOP_SECONDS, end the run.
                            if saw_any_event:
                                if below_since is None:
                                    below_since = now_pc
                                elif (now_pc - below_since) >= AUTO_STOP_SECONDS:
                                    self.gui_queue.put(
                                        (
                                            "status",
                                            f"AUTO-STOP: below {FORCE_THRESHOLD_N} N for {AUTO_STOP_SECONDS:.1f}s",
                                        )
                                    )
                                    break

                    else:
                        # We are inside an event: always log the sample.
                        writer.writerow([event_id, now_iso, t_elapsed, stm32_ms, force_N, line])
                        f.flush()

                        # End debounce: wait for END_BELOW_CYCLES consecutive samples
                        # below threshold, then start the POST_SAMPLES tail.
                        if post_remaining == 0:
                            if force_N < FORCE_THRESHOLD_N:
                                below_count += 1
                                if below_count >= END_BELOW_CYCLES:
                                    post_remaining = POST_SAMPLES
                                    below_since = now_pc
                                    self.gui_queue.put(
                                        (
                                            "status",
                                            f"EVENT {event_id} tail {POST_SAMPLES} samples "
                                            f"(below for {END_BELOW_CYCLES} cycles)",
                                        )
                                    )
                            else:
                                # Went back above threshold: cancel end-debounce.
                                below_count = 0

                        if post_remaining > 0:
                            post_remaining -= 1
                            if post_remaining == 0:
                                # Tail finished: event complete, go back to idle state.
                                in_event = False
                                pre_buffer.clear()
                                above_count = 0
                                below_count = 0
                                self.gui_queue.put(("status", f"EVENT {event_id} COMPLETE"))

                        # If force spikes back above threshold during the tail, cancel
                        # the below timer / end debounce.
                        if force_N >= FORCE_THRESHOLD_N:
                            below_since = None
                            below_count = 0

            # Stop audio + serial
            self.gui_queue.put(("status", "Stopping audio..."))
            audio.stop()
            ser.close()

            # Post-process combined CSV from FULL wav (keeps alignment to pc_elapsed_s).
            self.gui_queue.put(("status", "Post-processing: creating combined CSV (aligned to force)..."))
            postprocess_event_aligned(
                self.paths.raw_event_csv,
                self.paths.wav_full_path,
                self.paths.combined_csv,
                RMS_WINDOW_S,
            )

            # Save force/audio plot (from combined CSV).
            self.gui_queue.put(("status", "Saving force/audio plot PNG..."))
            save_force_audio_plot(self.paths.combined_csv, self.paths.plot_png)

            # Extract ONLY event audio to final WAV and delete the full recording.
            self.gui_queue.put(("status", "Extracting event-only audio WAV..."))
            intervals = get_event_intervals_from_raw_csv(self.paths.raw_event_csv)
            if not intervals:
                raise RuntimeError("No events found in raw CSV, so no event-only audio to save.")

            write_event_only_wav(self.paths.wav_full_path, self.paths.wav_event_path, intervals)

            # Compute spectrogram from event-only WAV.
            self.gui_queue.put(("status", "Computing spectrogram (event-only audio)..."))
            compute_audio_spectrogram(
                self.paths.wav_event_path,
                self.paths.spectrogram_csv,
                self.paths.spectrogram_png,
                win_s=SPEC_WIN_S,
                overlap=SPEC_OVERLAP,
                nfft=SPEC_NFFT,
                max_hz=SPEC_MAX_HZ,
            )

            # Delete the full wav so only event-only audio remains on disk.
            try:
                os.remove(self.paths.wav_full_path)
            except Exception:
                pass

            # Summarize outputs to the GUI.
            self.gui_queue.put(
                (
                    "done",
                    "Done.\n"
                    f"Raw CSV: {self.paths.raw_event_csv}\n"
                    f"WAV (event-only): {self.paths.wav_event_path}\n"
                    f"Combined (force-aligned): {self.paths.combined_csv}\n"
                    f"Force/Audio Plot: {self.paths.plot_png}\n"
                    f"Spectrogram CSV (event-only wav timebase): {self.paths.spectrogram_csv}\n"
                    f"Spectrogram Plot: {self.paths.spectrogram_png}\n",
                )
            )

        except Exception as e:
            # Any unhandled error is reported back to the GUI for display.
            self.gui_queue.put(("error", str(e)))


class App(tk.Tk):
    """
    Tkinter GUI for configuring and running the acquisition.

    Main elements:
      - Status line (top).
      - Start/Stop buttons.
      - Run name entry.
      - Read-only summary of key settings (for quick reference).
      - Live log window showing force samples and status messages.
    """

    def __init__(self):
        super().__init__()
        self.title("STM32 Load Cell + USB Mic Logger (Event Audio Only)")
        self.geometry("980x720")

        self.gui_queue = queue.Queue()
        self.worker = None
        self.current_paths = None

        self._build_ui()
        # Periodically poll the worker's message queue and update the UI.
        self.after(50, self._poll_queue)

    def _build_ui(self):
        """Construct all Tkinter widgets."""
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", fill="x", expand=True)

        self.start_btn = ttk.Button(top, text="Start Run", command=self.start_run)
        self.start_btn.pack(side="right", padx=(5, 0))

        self.stop_btn = ttk.Button(top, text="Stop Run", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="right", padx=(5, 0))

        # Run naming field (used as prefix before timestamp).
        runfrm = ttk.Frame(frm)
        runfrm.pack(fill="x", pady=(8, 0))
        ttk.Label(runfrm, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(value="loadcell_run")
        ttk.Entry(runfrm, textvariable=self.run_name_var, width=32).pack(side="left", padx=(6, 0))

        # Parameters display (read-only; edit constants in code if needed).
        params = ttk.LabelFrame(frm, text="Run Settings (edit in script constants)", padding=8)
        params.pack(fill="x", pady=10)

        ttk.Label(params, text=f"Threshold: {FORCE_THRESHOLD_N} N").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(
            params, text=f"Start debounce: {START_ABOVE_CYCLES} cycles >= threshold"
        ).grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(
            params, text=f"End debounce: {END_BELOW_CYCLES} cycles < threshold"
        ).grid(row=0, column=2, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Pre: {PRE_SAMPLES} samples").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Post: {POST_SAMPLES} samples").grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Auto-stop: {AUTO_STOP_SECONDS} s below").grid(
            row=1, column=2, sticky="w", padx=6, pady=2
        )

        ttk.Label(params, text=f"Audio: {AUDIO_FS} Hz, {AUDIO_CHANNELS} ch").grid(
            row=2, column=0, sticky="w", padx=6, pady=2
        )
        ttk.Label(params, text=f"RMS window: {RMS_WINDOW_S} s").grid(row=2, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Event audio pad: {EVENT_AUDIO_PAD_S:.2f} s").grid(
            row=3, column=0, sticky="w", padx=6, pady=2
        )
        ttk.Label(params, text=f"Merge gap: {MERGE_GAP_S:.2f} s").grid(row=3, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Spectrogram win: {SPEC_WIN_S:.3f} s").grid(
            row=4, column=0, sticky="w", padx=6, pady=2
        )
        ttk.Label(params, text=f"Spectrogram overlap: {int(SPEC_OVERLAP*100)}%").grid(
            row=4, column=1, sticky="w", padx=6, pady=2
        )
        ttk.Label(params, text=f"Spectrogram NFFT: {SPEC_NFFT}").grid(
            row=4, column=2, sticky="w", padx=6, pady=2
        )
        ttk.Label(
            params,
            text=f"Spectrogram max Hz: {SPEC_MAX_HZ if SPEC_MAX_HZ is not None else 'Nyquist'}",
        ).grid(row=5, column=0, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Output dir: {RUNS_DIR}/").grid(row=5, column=2, sticky="w", padx=6, pady=2)

        # Live log window: shows status and decoded STM32 lines.
        logfrm = ttk.LabelFrame(frm, text="Live Log", padding=8)
        logfrm.pack(fill="both", expand=True)

        self.text = tk.Text(logfrm, height=22, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(logfrm, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

    def start_run(self):
        """Create RunPaths, start a RunWorker, and update the UI state."""
        if self.worker is not None:
            return

        try:
            run_name = self.run_name_var.get()
            self.current_paths = make_run_paths(run_name)

            self.status_var.set(f"Preparing run: {self.current_paths.base_name}")
            self.text.insert("end", f"\n=== START RUN: {self.current_paths.base_name} ===\n")
            self.text.see("end")

            self.worker = RunWorker(self.gui_queue, self.current_paths, mic_device=None)
            self.worker.start()

            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Start Run Failed", str(e))
            self.worker = None

    def stop_run(self):
        """Request a graceful stop of the current run."""
        if self.worker is None:
            return
        self.status_var.set("Stop requested...")
        self.worker.request_stop()
        self.stop_btn.configure(state="disabled")

    def _poll_queue(self):
        """
        Periodically drain the GUI message queue and update widgets.

        Messages from RunWorker have one of the following types:
          - "status": short text for status line + log.
          - "line"  : raw sample/telemetry line for the log.
          - "done"  : run finished successfully.
          - "error" : run failed; show error dialog.
        """
        try:
            while True:
                msg_type, payload = self.gui_queue.get_nowait()

                if msg_type == "status":
                    self.status_var.set(payload)
                    self.text.insert("end", f"[STATUS] {payload}\n")
                    self.text.see("end")

                elif msg_type == "line":
                    self.text.insert("end", payload + "\n")
                    self.text.see("end")

                elif msg_type == "done":
                    self.status_var.set("Idle.")
                    self.text.insert("end", f"\n=== RUN COMPLETE ===\n{payload}\n")
                    self.text.see("end")
                    self._reset_buttons()

                elif msg_type == "error":
                    self.status_var.set("Error.")
                    self.text.insert("end", f"\n[ERROR] {payload}\n")
                    self.text.see("end")
                    messagebox.showerror("Run Error", payload)
                    self._reset_buttons()

        except queue.Empty:
            # No more messages at this moment; schedule the next poll.
            pass

        self.after(50, self._poll_queue)

    def _reset_buttons(self):
        """Return the Start/Stop buttons to their idle state."""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.worker = None


if __name__ == "__main__":
    ensure_dir(RUNS_DIR)
    app = App()
    app.mainloop()
