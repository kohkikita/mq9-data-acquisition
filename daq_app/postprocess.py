# daq_app/postprocess.py
import os
import math
import numpy as np
import pandas as pd
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    RMS_WINDOW_S, EPS, PLOT_DPI,
    SPEC_WIN_S, SPEC_OVERLAP, SPEC_NFFT, SPEC_MAX_HZ, SPEC_CMAP,
    EVENT_AUDIO_PAD_S, MERGE_GAP_S,
)


def postprocess_event_aligned(force_csv_path: str, wav_path: str, out_csv_path: str, rms_window_s: float = RMS_WINDOW_S) -> None:
    df = pd.read_csv(force_csv_path)
    if "pc_elapsed_s" not in df.columns or "force_N" not in df.columns:
        raise RuntimeError("Force CSV missing required columns: pc_elapsed_s, force_N")

    df = df.copy()
    df["pc_elapsed_s"] = pd.to_numeric(df["pc_elapsed_s"], errors="coerce").astype(float)
    df["force_N"] = pd.to_numeric(df["force_N"], errors="coerce").astype(float)
    df = df[np.isfinite(df["pc_elapsed_s"].values)]
    if df.empty:
        raise RuntimeError("Force CSV has no valid pc_elapsed_s rows.")
    df = df.sort_values("pc_elapsed_s").drop_duplicates(subset=["pc_elapsed_s"], keep="last")

    t_min = float(df["pc_elapsed_s"].min())
    t_max = float(df["pc_elapsed_s"].max())

    fs, audio = wavfile.read(wav_path)
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

    first_i = last_i = None
    for i in range(n_windows):
        t_mid = (i * rms_window_s) + 0.5 * rms_window_s
        if t_min <= t_mid <= t_max:
            first_i = i
            break
    for i in range(n_windows - 1, -1, -1):
        t_mid = (i * rms_window_s) + 0.5 * rms_window_s
        if t_min <= t_mid <= t_max:
            last_i = i
            break
    if first_i is None or last_i is None:
        raise RuntimeError("No audio windows overlap the force event window.")

    t0_event = first_i * rms_window_s

    t_src = df["pc_elapsed_s"].values
    force_src = df["force_N"].values

    vesc_cols = [
        "vesc_rpm",
        "vesc_v_in_V",
        "vesc_i_motor_A",
        "vesc_i_in_A",
        "vesc_duty",
        "vesc_temp_mos_C",
        "vesc_power_W",
    ]

    present_vesc = [c for c in vesc_cols if c in df.columns]
    for c in present_vesc:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    def interp_optional(col: str, t_query: float) -> float:
        y = df[col].values
        mask = np.isfinite(t_src) & np.isfinite(y)
        if mask.sum() < 2:
            return float("nan")
        return float(np.interp(t_query, t_src[mask], y[mask]))

    rows = []
    for i in range(first_i, last_i + 1):
        t_start = i * rms_window_s
        t_end = (i + 1) * rms_window_s
        t_mid = t_start + 0.5 * rms_window_s

        seg = audio[i * win_len:(i + 1) * win_len]
        rms_lin = float(np.sqrt(np.mean(seg ** 2)))
        rms_dbfs = float(20 * np.log10(rms_lin + EPS))

        force_N = float(np.interp(t_mid, t_src, force_src))

        out = {
            "t_event_s": t_mid - t0_event,
            "force_N": force_N,
            "audio_rms_dbfs": rms_dbfs,
            "audio_rms_linear": rms_lin,
            "window_start_s": t_start - t0_event,
            "window_end_s": t_end - t0_event,
        }
        for c in present_vesc:
            out[c] = interp_optional(c, t_mid)

        rows.append(out)

    df_out = pd.DataFrame(rows)
    df_out["t_event_s"] = df_out["t_event_s"].round(3)
    df_out["window_start_s"] = df_out["window_start_s"].round(3)
    df_out["window_end_s"] = df_out["window_end_s"].round(3)
    df_out["force_N"] = df_out["force_N"].round(3)
    df_out["audio_rms_dbfs"] = df_out["audio_rms_dbfs"].round(2)
    df_out["audio_rms_linear"] = df_out["audio_rms_linear"].round(6)

    for c in present_vesc:
        if "temp" in c:
            df_out[c] = df_out[c].round(1)
        elif c in ("vesc_v_in_V", "vesc_i_motor_A", "vesc_i_in_A"):
            df_out[c] = df_out[c].round(2)
        elif c == "vesc_duty":
            df_out[c] = df_out[c].round(4)
        else:
            df_out[c] = df_out[c].round(1)

    df_out.to_csv(out_csv_path, index=False)


def _add_right_axis(ax_base, offset_axes: float):
    ax = ax_base.twinx()
    ax.spines["right"].set_position(("axes", offset_axes))
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    return ax


def save_overlay_force_audio_rpm_power_duty(combined_csv_path: str, out_png_path: str) -> None:
    df = pd.read_csv(combined_csv_path)

    required = {"t_event_s", "force_N", "audio_rms_dbfs"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Combined CSV missing required columns: {sorted(required)}")

    t = df["t_event_s"].astype(float).values

    fig, ax_force = plt.subplots(figsize=(12, 6))
    ax_audio = ax_force.twinx()
    ax_rpm = _add_right_axis(ax_force, 1.10)
    ax_power = _add_right_axis(ax_force, 1.20)
    ax_duty = _add_right_axis(ax_force, 1.30)

    handles = []
    labels = []

    p1, = ax_force.plot(t, df["force_N"].astype(float).values, linewidth=1.9, color="tab:blue", label="Force (N)")
    ax_force.set_ylabel("Force (N)", color="tab:blue")
    ax_force.tick_params(axis="y", colors="tab:blue")
    handles.append(p1); labels.append("Force (N)")

    p2, = ax_audio.plot(t, df["audio_rms_dbfs"].astype(float).values, linewidth=1.6, color="tab:orange", label="Audio (dBFS)")
    ax_audio.set_ylabel("Audio (dBFS)", color="tab:orange")
    ax_audio.tick_params(axis="y", colors="tab:orange")
    handles.append(p2); labels.append("Audio (dBFS)")

    if "vesc_rpm" in df.columns:
        p3, = ax_rpm.plot(t, df["vesc_rpm"].astype(float).values, linewidth=1.5, color="tab:purple", label="RPM")
        ax_rpm.set_ylabel("RPM", color="tab:purple")
        ax_rpm.tick_params(axis="y", colors="tab:purple")
        handles.append(p3); labels.append("RPM")
    else:
        ax_rpm.set_visible(False)

    if "vesc_power_W" in df.columns:
        p4, = ax_power.plot(t, df["vesc_power_W"].astype(float).values, linewidth=1.5, color="tab:red", label="Power (W)")
        ax_power.set_ylabel("Power (W)", color="tab:red")
        ax_power.tick_params(axis="y", colors="tab:red")
        handles.append(p4); labels.append("Power (W)")
    else:
        ax_power.set_visible(False)

    if "vesc_duty" in df.columns:
        p5, = ax_duty.plot(t, df["vesc_duty"].astype(float).values, linewidth=1.5, color="tab:green", label="Duty")
        ax_duty.set_ylabel("Duty", color="tab:green")
        ax_duty.tick_params(axis="y", colors="tab:green")
        handles.append(p5); labels.append("Duty")
    else:
        ax_duty.set_visible(False)

    ax_force.set_title("Overlay: Force + Audio + RPM + Power + Duty")
    ax_force.set_xlabel("Time (s)")
    ax_force.grid(True, alpha=0.3)

    ax_force.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_overlay_force_audio_rpm(combined_csv_path: str, out_png_path: str) -> None:
    df = pd.read_csv(combined_csv_path)

    required = {"t_event_s", "force_N", "audio_rms_dbfs"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Combined CSV missing required columns: {sorted(required)}")
    if "vesc_rpm" not in df.columns:
        raise RuntimeError("Combined CSV missing vesc_rpm")

    t = df["t_event_s"].astype(float).values

    fig, ax_force = plt.subplots(figsize=(12, 6))
    ax_audio = ax_force.twinx()
    ax_rpm = _add_right_axis(ax_force, 1.10)

    handles = []
    labels = []

    p1, = ax_force.plot(t, df["force_N"].astype(float).values, linewidth=1.9, color="tab:blue", label="Force (N)")
    ax_force.set_ylabel("Force (N)", color="tab:blue")
    ax_force.tick_params(axis="y", colors="tab:blue")
    handles.append(p1); labels.append("Force (N)")

    p2, = ax_audio.plot(t, df["audio_rms_dbfs"].astype(float).values, linewidth=1.6, color="tab:orange", label="Audio (dBFS)")
    ax_audio.set_ylabel("Audio (dBFS)", color="tab:orange")
    ax_audio.tick_params(axis="y", colors="tab:orange")
    handles.append(p2); labels.append("Audio (dBFS)")

    p3, = ax_rpm.plot(t, df["vesc_rpm"].astype(float).values, linewidth=1.5, color="tab:purple", label="RPM")
    ax_rpm.set_ylabel("RPM", color="tab:purple")
    ax_rpm.tick_params(axis="y", colors="tab:purple")
    handles.append(p3); labels.append("RPM")

    ax_force.set_title("Overlay: Force + Audio + RPM")
    ax_force.set_xlabel("Time (s)")
    ax_force.grid(True, alpha=0.3)

    ax_force.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_overlay_force_audio_power(combined_csv_path: str, out_png_path: str) -> None:
    df = pd.read_csv(combined_csv_path)

    required = {"t_event_s", "force_N", "audio_rms_dbfs"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Combined CSV missing required columns: {sorted(required)}")
    if "vesc_power_W" not in df.columns:
        raise RuntimeError("Combined CSV missing vesc_power_W")

    t = df["t_event_s"].astype(float).values

    fig, ax_force = plt.subplots(figsize=(12, 6))
    ax_audio = ax_force.twinx()
    ax_power = _add_right_axis(ax_force, 1.10)

    handles = []
    labels = []

    p1, = ax_force.plot(t, df["force_N"].astype(float).values, linewidth=1.9, color="tab:blue", label="Force (N)")
    ax_force.set_ylabel("Force (N)", color="tab:blue")
    ax_force.tick_params(axis="y", colors="tab:blue")
    handles.append(p1); labels.append("Force (N)")

    p2, = ax_audio.plot(t, df["audio_rms_dbfs"].astype(float).values, linewidth=1.6, color="tab:orange", label="Audio (dBFS)")
    ax_audio.set_ylabel("Audio (dBFS)", color="tab:orange")
    ax_audio.tick_params(axis="y", colors="tab:orange")
    handles.append(p2); labels.append("Audio (dBFS)")

    p3, = ax_power.plot(t, df["vesc_power_W"].astype(float).values, linewidth=1.5, color="tab:red", label="Power (W)")
    ax_power.set_ylabel("Power (W)", color="tab:red")
    ax_power.tick_params(axis="y", colors="tab:red")
    handles.append(p3); labels.append("Power (W)")

    ax_force.set_title("Overlay: Force + Audio + Power")
    ax_force.set_xlabel("Time (s)")
    ax_force.grid(True, alpha=0.3)

    ax_force.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)


def get_event_intervals_from_raw_csv(raw_event_csv: str) -> list[tuple[float, float]]:
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
        a = float(s.min())  # pad handled via EVENT_AUDIO_PAD_S below
        b = float(s.max())
        a -= EVENT_AUDIO_PAD_S
        b += EVENT_AUDIO_PAD_S
        intervals.append((max(0.0, a), max(0.0, b)))

    intervals.sort(key=lambda x: x[0])

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


def write_event_only_wav(full_wav_path: str, out_wav_path: str, intervals_s: list[tuple[float, float]]) -> None:
    if not os.path.exists(full_wav_path):
        raise RuntimeError(f"Full WAV not found: {full_wav_path}")

    fs, audio = wavfile.read(full_wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

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


def compute_audio_spectrogram(
    wav_path: str,
    out_csv_path: str,
    out_png_path: str,
    *,
    win_s: float = SPEC_WIN_S,
    overlap: float = SPEC_OVERLAP,
    nfft: int = SPEC_NFFT,
    max_hz: float | None = SPEC_MAX_HZ
) -> None:
    if not os.path.exists(wav_path):
        raise RuntimeError(f"WAV not found: {wav_path}")

    fs, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

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

    if nfft < win_len:
        nfft = int(2 ** math.ceil(math.log2(win_len)))

    hop = int(round(win_len * (1.0 - overlap)))
    hop = max(1, hop)

    window = np.hanning(win_len)

    n_frames = 1 + (audio.size - win_len) // hop if audio.size >= win_len else 0
    if n_frames <= 0:
        raise RuntimeError("Audio too short for chosen spectrogram window.")

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    fmask = freqs <= float(max_hz) if max_hz is not None else slice(None)
    freqs_use = freqs[fmask]

    spec_db = np.empty((freqs_use.size, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        seg = audio[start:start + win_len]
        segw = seg * window
        X = np.fft.rfft(segw, n=nfft)
        P = (np.abs(X) ** 2)
        P = P[fmask]
        spec_db[:, i] = 10.0 * np.log10(np.maximum(P, 1e-20))

    t_s = (np.arange(n_frames) * hop + (win_len / 2.0)) / fs

    chunks = []
    for ti in range(n_frames):
        chunks.append(pd.DataFrame({
            "t_s": np.full(freqs_use.size, t_s[ti], dtype=np.float64),
            "freq_hz": freqs_use.astype(np.float64),
            "mag_db": spec_db[:, ti].astype(np.float64),
        }))
    df = pd.concat(chunks, ignore_index=True)
    df["t_s"] = df["t_s"].round(4)
    df["freq_hz"] = df["freq_hz"].round(2)
    df["mag_db"] = df["mag_db"].round(2)
    df.to_csv(out_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    im = ax.imshow(
        spec_db, origin="lower", aspect="auto",
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
