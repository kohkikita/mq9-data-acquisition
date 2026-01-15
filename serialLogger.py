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

# Plotting (headless-safe for Tkinter apps on Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
BAUD = 115200
SERIAL_TIMEOUT_S = 1.0

# Trigger / capture behavior (event-only logging)
FORCE_THRESHOLD_N = 0.4
PRE_SAMPLES = 25
POST_SAMPLES = 25
AUTO_STOP_SECONDS = 1.0  # after event: auto-stop after this long continuously below threshold

# Debounced start/end conditions
START_ABOVE_CYCLES = 15  # require N consecutive samples >= threshold to start event
END_BELOW_CYCLES = 15    # require N consecutive samples < threshold to end event

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
SPEC_WIN_S = 0.050          # window length in seconds (e.g., 50 ms)
SPEC_OVERLAP = 0.75         # overlap fraction (0.0 to <1.0), e.g., 75%
SPEC_NFFT = 4096            # FFT size (>= window length in samples)
SPEC_MAX_HZ = None          # e.g., 10000 to limit to 10 kHz
SPEC_CMAP = "magma"

# Audio trimming behavior
EVENT_AUDIO_PAD_S = 0.05    # pad each event interval on both sides when extracting audio
MERGE_GAP_S = 0.02          # merge adjacent event intervals if gap <= this
# ------------------------------------------------

# Matches STM32 lines like: "Load=0.123 N, t=4567 ms"
LINE_RE = re.compile(r"Load=([+-]?\d+(?:\.\d+)?)\s*N,\s*t=(\d+)\s*ms")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sanitize_run_name(name: str) -> str:
    """
    Make a filesystem-safe run name:
      - trims whitespace
      - spaces -> underscores
      - keeps only letters/numbers/_/-
      - falls back to "run" if empty
    """
    name = (name or "").strip()
    if not name:
        return "run"
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "run"


def find_stm32_port() -> str:
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

    # Otherwise ambiguous
    lines = ["Could not uniquely identify STM32 serial port.", "Available ports:"]
    for p in ports:
        lines.append(f"  {p.device}: {p.description} ({p.manufacturer})")
    raise RuntimeError("\n".join(lines))


def parse_line(line: str):
    m = LINE_RE.search(line)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


@dataclass
class RunPaths:
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
    Output naming:
      <run_name>_<YYYY-MM-DD>_<HH-MM-SS>.*
    Timestamp remains at the END of the prefix.
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


def postprocess_event_aligned(force_csv_path: str, wav_path: str, out_csv_path: str,
                              rms_window_s: float = RMS_WINDOW_S) -> None:
    """
    Produces an event-aligned combined CSV (audio RMS windows aligned to force).
    Uses pc_elapsed_s in the force CSV as the shared timebase (relative to run start).
    Output columns are numeric and rounded.

    IMPORTANT: This expects wav_path timebase to match pc_elapsed_s (i.e., audio starts near run start).
               Therefore we run this on the FULL wav, not the trimmed event-only wav.
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

    # Determine the first/last audio window with midpoint inside the force window
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

    # Window-aligned event t0 (first included window start)
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

    df["t_event_s"] = df["t_event_s"].round(3)
    df["window_start_s"] = df["window_start_s"].round(3)
    df["window_end_s"] = df["window_end_s"].round(3)
    df["force_N"] = df["force_N"].round(3)
    df["audio_rms_dbfs"] = df["audio_rms_dbfs"].round(2)
    df["audio_rms_linear"] = df["audio_rms_linear"].round(6)

    df.to_csv(out_csv_path, index=False)


def save_force_audio_plot(combined_csv_path: str, out_png_path: str,
                          title: str = "Force vs Audio RMS (Post-Processed)") -> None:
    """
    Reads the combined_event_aligned CSV and saves a PNG plot:
      - Audio RMS (dBFS) on left axis
      - Force (N) on right axis

    UPDATED: force and audio are explicitly colored and legend is added.
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

    # Explicit colors + labels
    ax_audio.plot(t, audio_db, color="tab:orange", linewidth=1.8, label="Audio (dBFS)")
    ax_force.plot(t, force, color="tab:blue", linewidth=1.8, label="Force (N)")

    ax_audio.set_title(title)
    ax_audio.set_xlabel("Time (s)")

    ax_audio.set_ylabel("Audio Level (dBFS)", color="tab:orange")
    ax_force.set_ylabel("Force (N)", color="tab:blue")

    ax_audio.tick_params(axis="y", colors="tab:orange")
    ax_force.tick_params(axis="y", colors="tab:blue")

    ax_audio.grid(True, which="both", alpha=0.3)

    # Combined legend
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
    Computes STFT spectrogram and saves:
      - CSV (long format): t_s, freq_hz, mag_db
      - PNG: time-frequency heatmap

    mag_db is relative (not calibrated SPL).
    Time axis is relative to the provided wav_path (for event-only wav, this is "event-audio timeline").
    """
    if not os.path.exists(wav_path):
        raise RuntimeError(f"WAV not found: {wav_path}")

    fs, audio = wavfile.read(wav_path)

    # Mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Normalize to float [-1, 1] when integer PCM
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float64) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float64)

    audio = audio[np.isfinite(audio)]
    if audio.size == 0:
        raise RuntimeError("Audio empty.")

    win_len = int(round(win_s * fs))
    if win_len < 16:
        raise RuntimeError("SPEC_WIN_S too small (window < 16 samples).")

    # enforce nfft >= win_len
    if nfft < win_len:
        nfft = int(2 ** math.ceil(math.log2(win_len)))

    hop = int(round(win_len * (1.0 - overlap)))
    hop = max(1, hop)

    window = np.hanning(win_len)

    # Number of frames
    n_frames = 1 + (audio.size - win_len) // hop if audio.size >= win_len else 0
    if n_frames <= 0:
        raise RuntimeError("Audio too short for chosen spectrogram window.")

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Optional frequency limit
    if max_hz is not None:
        fmask = freqs <= float(max_hz)
    else:
        fmask = slice(None)

    freqs_use = freqs[fmask]

    # spec_db is [freq_bins, time_frames]
    spec_db = np.empty((freqs_use.size, n_frames), dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        seg = audio[start:start + win_len]
        segw = seg * window

        X = np.fft.rfft(segw, n=nfft)
        P = (np.abs(X) ** 2)  # power (relative)

        P = P[fmask]
        spec_db[:, i] = 10.0 * np.log10(np.maximum(P, 1e-20))

    # Time axis in seconds (center of window)
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
    Reads the raw event CSV and returns merged event intervals in pc_elapsed_s.

    We treat each event_id >= 1 as an interval from min(pc_elapsed_s) to max(pc_elapsed_s),
    then:
      - pad each interval by EVENT_AUDIO_PAD_S
      - merge intervals separated by <= MERGE_GAP_S
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

    # Merge close/overlapping
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
    Reads full_wav_path and writes out_wav_path containing only samples within intervals_s,
    concatenated back-to-back.

    The resulting time axis is a concatenated "event-audio timeline" (gaps removed).
    """
    if not os.path.exists(full_wav_path):
        raise RuntimeError(f"Full WAV not found: {full_wav_path}")

    fs, audio = wavfile.read(full_wav_path)

    if audio.ndim > 1:
        audio = audio[:, 0]

    # Ensure we output int16 PCM
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
    Records WAV to disk using sounddevice + soundfile in a callback stream.
    """

    def __init__(self, wav_path: str, fs: int, channels: int, device=None):
        self.wav_path = wav_path
        self.fs = fs
        self.channels = channels
        self.device = device

        self._sf = None
        self._stream = None

    def start(self):
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
    Background worker that:
      - reads serial lines (force)
      - implements event-only capture w/ pre/post/threshold/autostop + debounce
      - records FULL WAV temporarily
      - writes raw event CSV during run
      - on completion:
           * creates combined event-aligned CSV + force/audio plot from FULL WAV
           * extracts ONLY event audio to final WAV
           * computes spectrogram from event-only WAV
           * deletes FULL WAV so only event audio is kept
    """

    def __init__(self, gui_queue: queue.Queue, paths: RunPaths, mic_device=None):
        super().__init__(daemon=True)
        self.gui_queue = gui_queue
        self.paths = paths
        self.mic_device = mic_device
        self._stop_req = threading.Event()

    def request_stop(self):
        self._stop_req.set()

    def run(self):
        try:
            port = find_stm32_port()
            self.gui_queue.put(("status", f"Serial: {port} @ {BAUD}"))

            ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT_S)
            time.sleep(2.0)  # STM32 often resets on open

            # Use a single time origin for the run (pc_elapsed_s)
            t0 = time.perf_counter()

            # Start FULL audio (temporary)
            self.gui_queue.put(("status", "Starting audio (temporary full recording)..."))
            audio = AudioRecorder(self.paths.wav_full_path, AUDIO_FS, AUDIO_CHANNELS, device=self.mic_device)
            audio.start()

            # Event capture state
            pre_buffer = deque(maxlen=PRE_SAMPLES)
            in_event = False
            post_remaining = 0
            event_id = 0
            saw_any_event = False
            below_since = None  # PC clock (perf_counter)

            # Debounce counters
            above_count = 0  # consecutive >= threshold (start debounce)
            below_count = 0  # consecutive < threshold (end debounce)

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

                    self.gui_queue.put(("line", f"t={stm32_ms:8d} ms | pc={t_elapsed:8.3f} s | F={force_N:7.3f} N"))

                    row = [event_id, now_iso, t_elapsed, stm32_ms, force_N, line]
                    now_pc = time.perf_counter()

                    if not in_event:
                        pre_buffer.append(row)

                        # Start debounce
                        if force_N >= FORCE_THRESHOLD_N:
                            above_count += 1
                            if above_count >= START_ABOVE_CYCLES:
                                in_event = True
                                post_remaining = 0
                                event_id += 1
                                saw_any_event = True
                                below_since = None

                                below_count = 0
                                self.gui_queue.put(("status", f"EVENT {event_id} START (above for {START_ABOVE_CYCLES} cycles)"))

                                # Flush pre-buffer as part of event
                                for r in pre_buffer:
                                    r[0] = event_id
                                    writer.writerow(r)
                                f.flush()
                                pre_buffer.clear()

                                writer.writerow([event_id, now_iso, t_elapsed, stm32_ms, force_N, line])
                                f.flush()
                        else:
                            above_count = 0

                            # Auto-stop after event happened AND continuously below threshold
                            if saw_any_event:
                                if below_since is None:
                                    below_since = now_pc
                                elif (now_pc - below_since) >= AUTO_STOP_SECONDS:
                                    self.gui_queue.put(("status", f"AUTO-STOP: below {FORCE_THRESHOLD_N} N for {AUTO_STOP_SECONDS:.1f}s"))
                                    break

                    else:
                        # In-event logging
                        writer.writerow([event_id, now_iso, t_elapsed, stm32_ms, force_N, line])
                        f.flush()

                        # End debounce: require END_BELOW_CYCLES consecutive samples < threshold before tail
                        if post_remaining == 0:
                            if force_N < FORCE_THRESHOLD_N:
                                below_count += 1
                                if below_count >= END_BELOW_CYCLES:
                                    post_remaining = POST_SAMPLES
                                    below_since = now_pc
                                    self.gui_queue.put(("status", f"EVENT {event_id} tail {POST_SAMPLES} samples (below for {END_BELOW_CYCLES} cycles)"))
                            else:
                                below_count = 0

                        if post_remaining > 0:
                            post_remaining -= 1
                            if post_remaining == 0:
                                in_event = False
                                pre_buffer.clear()
                                above_count = 0
                                below_count = 0
                                self.gui_queue.put(("status", f"EVENT {event_id} COMPLETE"))

                        # If force goes back above threshold, cancel below timer and reset end debounce
                        if force_N >= FORCE_THRESHOLD_N:
                            below_since = None
                            below_count = 0

            # Stop audio + serial
            self.gui_queue.put(("status", "Stopping audio..."))
            audio.stop()
            ser.close()

            # Post-process combined CSV from FULL wav (keeps alignment to pc_elapsed_s)
            self.gui_queue.put(("status", "Post-processing: creating combined CSV (aligned to force)..."))
            postprocess_event_aligned(self.paths.raw_event_csv, self.paths.wav_full_path, self.paths.combined_csv, RMS_WINDOW_S)

            # Save force/audio plot (from combined CSV)
            self.gui_queue.put(("status", "Saving force/audio plot PNG..."))
            save_force_audio_plot(self.paths.combined_csv, self.paths.plot_png)

            # Extract ONLY event audio to final WAV and delete the full recording
            self.gui_queue.put(("status", "Extracting event-only audio WAV..."))
            intervals = get_event_intervals_from_raw_csv(self.paths.raw_event_csv)
            if not intervals:
                raise RuntimeError("No events found in raw CSV, so no event-only audio to save.")

            write_event_only_wav(self.paths.wav_full_path, self.paths.wav_event_path, intervals)

            # Compute spectrogram from event-only WAV
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

            # Delete the full wav so only event-only audio remains
            try:
                os.remove(self.paths.wav_full_path)
            except Exception:
                pass

            self.gui_queue.put((
                "done",
                "Done.\n"
                f"Raw CSV: {self.paths.raw_event_csv}\n"
                f"WAV (event-only): {self.paths.wav_event_path}\n"
                f"Combined (force-aligned): {self.paths.combined_csv}\n"
                f"Force/Audio Plot: {self.paths.plot_png}\n"
                f"Spectrogram CSV (event-only wav timebase): {self.paths.spectrogram_csv}\n"
                f"Spectrogram Plot: {self.paths.spectrogram_png}\n"
            ))

        except Exception as e:
            self.gui_queue.put(("error", str(e)))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STM32 Load Cell + USB Mic Logger (Event Audio Only)")
        self.geometry("980x720")

        self.gui_queue = queue.Queue()
        self.worker = None
        self.current_paths = None

        self._build_ui()
        self.after(50, self._poll_queue)

    def _build_ui(self):
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

        # Run naming field
        runfrm = ttk.Frame(frm)
        runfrm.pack(fill="x", pady=(8, 0))
        ttk.Label(runfrm, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(value="loadcell_run")
        ttk.Entry(runfrm, textvariable=self.run_name_var, width=32).pack(side="left", padx=(6, 0))

        # Parameters display
        params = ttk.LabelFrame(frm, text="Run Settings (edit in script constants)", padding=8)
        params.pack(fill="x", pady=10)

        ttk.Label(params, text=f"Threshold: {FORCE_THRESHOLD_N} N").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Start debounce: {START_ABOVE_CYCLES} cycles >= threshold").grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"End debounce: {END_BELOW_CYCLES} cycles < threshold").grid(row=0, column=2, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Pre: {PRE_SAMPLES} samples").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Post: {POST_SAMPLES} samples").grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Auto-stop: {AUTO_STOP_SECONDS} s below").grid(row=1, column=2, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Audio: {AUDIO_FS} Hz, {AUDIO_CHANNELS} ch").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"RMS window: {RMS_WINDOW_S} s").grid(row=2, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Event audio pad: {EVENT_AUDIO_PAD_S:.2f} s").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Merge gap: {MERGE_GAP_S:.2f} s").grid(row=3, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Spectrogram win: {SPEC_WIN_S:.3f} s").grid(row=4, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Spectrogram overlap: {int(SPEC_OVERLAP*100)}%").grid(row=4, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Spectrogram NFFT: {SPEC_NFFT}").grid(row=4, column=2, sticky="w", padx=6, pady=2)
        ttk.Label(params, text=f"Spectrogram max Hz: {SPEC_MAX_HZ if SPEC_MAX_HZ is not None else 'Nyquist'}").grid(row=5, column=0, sticky="w", padx=6, pady=2)

        ttk.Label(params, text=f"Output dir: {RUNS_DIR}/").grid(row=5, column=2, sticky="w", padx=6, pady=2)

        # Log window
        logfrm = ttk.LabelFrame(frm, text="Live Log", padding=8)
        logfrm.pack(fill="both", expand=True)

        self.text = tk.Text(logfrm, height=22, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(logfrm, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

    def start_run(self):
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
        if self.worker is None:
            return
        self.status_var.set("Stop requested...")
        self.worker.request_stop()
        self.stop_btn.configure(state="disabled")

    def _poll_queue(self):
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
            pass

        self.after(50, self._poll_queue)

    def _reset_buttons(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.worker = None


if __name__ == "__main__":
    ensure_dir(RUNS_DIR)
    app = App()
    app.mainloop()
