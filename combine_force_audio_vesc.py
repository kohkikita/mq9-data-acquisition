import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# VESC parsing assumptions
# ----------------------------
VESC_COLS = [
    "vesc_time_s",
    "vesc_power_W",
    "vesc_current_motor_A",
    "vesc_current_in_A",
    "vesc_voltage_in_V",
    "vesc_erpm",
    "vesc_temp_mosfet_C",
    "vesc_temp_motor_C",
    "vesc_duty_cycle_pct",
    "vesc_extra_9",
    "vesc_extra_10",
    "vesc_extra_11",
]


def read_force_audio_event_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"t_event_s", "force_N", "audio_rms_dbfs"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Force/Audio CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["t_event_s"] = df["t_event_s"].astype(float)
    df["force_N"] = df["force_N"].astype(float)
    df["audio_rms_dbfs"] = df["audio_rms_dbfs"].astype(float)
    df = df.sort_values("t_event_s").reset_index(drop=True)
    return df


def read_vesc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", header=None)

    if df.shape[1] != len(VESC_COLS):
        raise RuntimeError(
            f"Unexpected VESC column count: got {df.shape[1]}, expected {len(VESC_COLS)}.\n"
            "If your VESC export format differs, update VESC_COLS mapping."
        )

    df.columns = VESC_COLS
    df = df.copy()

    df["vesc_time_s"] = df["vesc_time_s"].astype(float)
    df["vesc_erpm"] = df["vesc_erpm"].astype(float)
    df["vesc_duty_cycle_pct"] = df["vesc_duty_cycle_pct"].astype(float)

    df = df.sort_values("vesc_time_s").reset_index(drop=True)

    # --- FIX: remove unused all-zero columns ---
    df = df.loc[:, (df != 0).any(axis=0)]

    return df


def _estimate_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return float(np.median(dt))


def find_drop_time_force_audio(
    t: np.ndarray,
    force: np.ndarray,
    audio_dbfs: np.ndarray,
    *,
    force_drop_frac: float,
    audio_drop_db: float,
    sustain_s: float,
    search_last_frac: float,
    pre_drop_offset_s: float = 0.0,   # NEW: shift earlier by this many seconds
) -> float:
    """
    Returns a timestamp JUST BEFORE the sustained drop region begins.

    We search in the last portion of the event (search_last_frac).
    We detect the earliest run where BOTH force/audio are "low" for sustain_s,
    then return the time right before the run starts (falling edge).
    """

    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)
    audio_dbfs = np.asarray(audio_dbfs, dtype=float)

    order = np.argsort(t)
    t = t[order]
    force = force[order]
    audio_dbfs = audio_dbfs[order]

    if len(t) < 10:
        return float(t[-1])

    dt = _estimate_dt(t)
    if not np.isfinite(dt) or dt <= 0:
        return float(t[-1])

    force_peak = float(np.nanmax(force))
    audio_peak = float(np.nanmax(audio_dbfs))

    force_thresh = force_drop_frac * force_peak
    audio_thresh = audio_peak - audio_drop_db

    start_i = int(np.floor((1.0 - search_last_frac) * len(t)))
    start_i = max(0, min(start_i, len(t) - 1))

    sustain_n = max(1, int(round(sustain_s / dt)))

    cond = (force <= force_thresh) & (audio_dbfs <= audio_thresh)
    cond = cond & np.isfinite(force) & np.isfinite(audio_dbfs)

    run = 0
    for i in range(start_i, len(cond)):
        if cond[i]:
            run += 1
            if run >= sustain_n:
                idx0 = i - sustain_n + 1  # start index of sustained low region

                # Return JUST BEFORE the low region begins
                edge_idx = max(start_i, idx0 - 7)

                t_edge = float(t[edge_idx])

                # Optional: shift a bit earlier (useful if your windowing lags)
                t_edge -= float(pre_drop_offset_s)

                # Clamp to data range
                if t_edge < float(t[0]):
                    t_edge = float(t[0])
                if t_edge > float(t[-1]):
                    t_edge = float(t[-1])

                return t_edge
        else:
            run = 0

    return float(t[-1])



def save_plot(df: pd.DataFrame, out_png: str, title: str) -> None:
    x = df["t_event_s"].astype(float)

    fig, ax_force = plt.subplots(figsize=(12, 6))

    ax_force.plot(x, df["force_N"].astype(float), color="blue", label="Force (N)")
    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("Force (N)", color="blue")
    ax_force.tick_params(axis="y", colors="blue")
    ax_force.grid(True, alpha=0.3)

    ax_audio = ax_force.twinx()
    ax_audio.plot(x, df["audio_rms_dbfs"].astype(float), color="red", label="Audio RMS (dBFS)")
    ax_audio.set_ylabel("Audio (dBFS)", color="red")
    ax_audio.tick_params(axis="y", colors="red")

    # RPM
    ax_rpm = ax_force.twinx()
    ax_rpm.spines["right"].set_position(("outward", 60))
    ax_rpm.plot(x, df["rpm_mech"].astype(float), color="green", label="RPM")
    ax_rpm.set_ylabel("RPM", color="green")
    ax_rpm.tick_params(axis="y", colors="green")

    lines = ax_force.get_lines() + ax_audio.get_lines() + ax_rpm.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax_force.legend(lines, labels, loc="upper left")

    ax_force.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Combine force/audio CSV with VESC CSV by aligning VESC last sample to force+audio drop-off."
    )
    ap.add_argument("--force_audio_csv", required=True, help="Path to *_combined_event_aligned.csv")
    ap.add_argument("--vesc_csv", required=True, help="Path to VESC export CSV (semicolon-delimited)")
    ap.add_argument("--out_csv", default="combined_event_aligned_with_vesc.csv", help="Output CSV filename")
    ap.add_argument("--out_png", default="combined_event_aligned_with_vesc.png", help="Output plot PNG filename")

    ap.add_argument("--pole_pairs", type=float, default=7.0, help="Motor pole pairs (ERPM / pole_pairs = mechanical RPM)")
    ap.add_argument("--merge_tolerance_s", type=float, default=0.10, help="Max time difference allowed when joining (seconds)")

    # Drop detection knobs
    ap.add_argument("--force_drop_frac", type=float, default=0.15, help="Force drop threshold as fraction of peak force")
    ap.add_argument("--audio_drop_db", type=float, default=8.0, help="Audio drop threshold: peak - this dBFS")
    ap.add_argument("--drop_sustain_s", type=float, default=0.5, help="Drop must hold for this long (seconds)")
    ap.add_argument("--search_last_frac", type=float, default=0.40, help="Search only in last fraction of force/audio window")

    args = ap.parse_args()

    if args.pole_pairs <= 0:
        raise RuntimeError("--pole_pairs must be > 0")

    df_force = read_force_audio_event_csv(args.force_audio_csv)
    df_vesc = read_vesc_csv(args.vesc_csv)

    # Mechanical RPM
    df_vesc = df_vesc.copy()
    df_vesc["rpm_mech"] = (df_vesc["vesc_erpm"] / args.pole_pairs).round().astype(int)

    # VESC "end" time = last data point
    t_vesc_end_abs = float(df_vesc["vesc_time_s"].iloc[-1])

    # Force/audio drop time (within last part of event)
    t_force_drop = find_drop_time_force_audio(
        t=df_force["t_event_s"].to_numpy(),
        force=df_force["force_N"].to_numpy(),
        audio_dbfs=df_force["audio_rms_dbfs"].to_numpy(),
        force_drop_frac=args.force_drop_frac,
        audio_drop_db=args.audio_drop_db,
        sustain_s=args.drop_sustain_s,
        search_last_frac=args.search_last_frac,
    )

    # Shift so VESC end aligns to force/audio drop
    shift = t_force_drop - t_vesc_end_abs

    df_vesc["t_event_s"] = df_vesc["vesc_time_s"] + shift
    df_vesc = df_vesc.sort_values("t_event_s").reset_index(drop=True)

    # Clip VESC to force window +/- pad
    tmin = float(df_force["t_event_s"].min())
    tmax = float(df_force["t_event_s"].max())
    df_vesc_clip = df_vesc[(df_vesc["t_event_s"] >= tmin - 3.0) & (df_vesc["t_event_s"] <= tmax + 3.0)].copy()
    df_vesc_clip = df_vesc_clip.sort_values("t_event_s").reset_index(drop=True)

    # Nearest join onto force/audio grid
    combined = pd.merge_asof(
        df_force.sort_values("t_event_s"),
        df_vesc_clip.sort_values("t_event_s"),
        on="t_event_s",
        direction="nearest",
        tolerance=args.merge_tolerance_s,
    )

    combined.to_csv(args.out_csv, index=False)

    save_plot(
        combined,
        out_png=args.out_png,
        title="End-Aligned: VESC last sample -> Force+Audio drop"
    )

    print("Wrote CSV:", args.out_csv)
    print("Wrote PNG:", args.out_png)
    print(f"VESC end time (last row):      vesc_time_s = {t_vesc_end_abs:.3f} s")
    print(f"Force+audio drop marker:       t_event_s   = {t_force_drop:.3f} s")
    print(f"Applied shift (VESC -> event): {shift:+.3f} s")
    print(f"Merge tolerance:              {args.merge_tolerance_s:.3f} s")


if __name__ == "__main__":
    main()
