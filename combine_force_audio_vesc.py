import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
import re

# Optional GUI (built-in)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


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

    # remove unused all-zero columns
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
    pre_drop_offset_s: float = 0.0,
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
                idx0 = i - sustain_n + 1
                edge_idx = max(start_i, idx0 - 7)

                t_edge = float(t[edge_idx])
                t_edge -= float(pre_drop_offset_s)

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


# ----------------------------
# NEW: run-name → safe filenames
# ----------------------------
def _sanitize_run_name(name: str) -> str:
    """
    Keep filenames portable:
    - allow letters/numbers/_/-
    - collapse whitespace to _
    - remove everything else
    """
    name = (name or "").strip()
    if not name:
        return "run"
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "run"


def make_output_paths(run_name: str, out_dir: str | Path | None) -> tuple[str, str]:
    run = _sanitize_run_name(run_name)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # date/time at END
    base = f"{run}_{ts}"

    out_dir = Path(out_dir) if out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    return str(out_dir / f"{base}.csv"), str(out_dir / f"{base}.png")


def process(
    *,
    force_audio_csv: str,
    vesc_csv: str,
    out_csv: str,
    out_png: str,
    pole_pairs: float,
    merge_tolerance_s: float,
    force_drop_frac: float,
    audio_drop_db: float,
    drop_sustain_s: float,
    search_last_frac: float,
) -> None:
    if pole_pairs <= 0:
        raise RuntimeError("--pole_pairs must be > 0")

    df_force = read_force_audio_event_csv(force_audio_csv)
    df_vesc = read_vesc_csv(vesc_csv)

    df_vesc = df_vesc.copy()
    df_vesc["rpm_mech"] = (df_vesc["vesc_erpm"] / pole_pairs).round().astype(int)

    t_vesc_end_abs = float(df_vesc["vesc_time_s"].iloc[-1])

    t_force_drop = find_drop_time_force_audio(
        t=df_force["t_event_s"].to_numpy(),
        force=df_force["force_N"].to_numpy(),
        audio_dbfs=df_force["audio_rms_dbfs"].to_numpy(),
        force_drop_frac=force_drop_frac,
        audio_drop_db=audio_drop_db,
        sustain_s=drop_sustain_s,
        search_last_frac=search_last_frac,
    )

    shift = t_force_drop - t_vesc_end_abs

    df_vesc["t_event_s"] = df_vesc["vesc_time_s"] + shift
    df_vesc = df_vesc.sort_values("t_event_s").reset_index(drop=True)

    tmin = float(df_force["t_event_s"].min())
    tmax = float(df_force["t_event_s"].max())
    df_vesc_clip = df_vesc[(df_vesc["t_event_s"] >= tmin - 3.0) & (df_vesc["t_event_s"] <= tmax + 3.0)].copy()
    df_vesc_clip = df_vesc_clip.sort_values("t_event_s").reset_index(drop=True)

    combined = pd.merge_asof(
        df_force.sort_values("t_event_s"),
        df_vesc_clip.sort_values("t_event_s"),
        on="t_event_s",
        direction="nearest",
        tolerance=merge_tolerance_s,
    )

    combined.to_csv(out_csv, index=False)

    save_plot(
        combined,
        out_png=out_png,
        title="End-Aligned: VESC last sample -> Force+Audio drop",
    )

    print("Wrote CSV:", out_csv)
    print("Wrote PNG:", out_png)
    print(f"VESC end time (last row):      vesc_time_s = {t_vesc_end_abs:.3f} s")
    print(f"Force+audio drop marker:       t_event_s   = {t_force_drop:.3f} s")
    print(f"Applied shift (VESC -> event): {shift:+.3f} s")
    print(f"Merge tolerance:              {merge_tolerance_s:.3f} s")


# ----------------------------
# NEW: Minimal GUI wrapper
# ----------------------------
def launch_gui() -> None:
    root = tk.Tk()
    root.title("DAQ Combine/Align Tool")

    pad = {"padx": 10, "pady": 6}

    # Vars
    force_var = tk.StringVar()
    vesc_var = tk.StringVar()
    run_var = tk.StringVar(value="run")
    outdir_var = tk.StringVar(value=str(Path.cwd()))

    pole_pairs_var = tk.DoubleVar(value=7.0)
    merge_tol_var = tk.DoubleVar(value=0.10)
    force_drop_frac_var = tk.DoubleVar(value=0.15)
    audio_drop_db_var = tk.DoubleVar(value=8.0)
    drop_sustain_var = tk.DoubleVar(value=0.5)
    search_last_frac_var = tk.DoubleVar(value=0.40)

    def browse_force():
        p = filedialog.askopenfilename(title="Select force/audio CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            force_var.set(p)

    def browse_vesc():
        p = filedialog.askopenfilename(title="Select VESC CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            vesc_var.set(p)

    def browse_outdir():
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            outdir_var.set(p)

    def run():
        try:
            if not force_var.get().strip():
                raise RuntimeError("Select a force/audio CSV.")
            if not vesc_var.get().strip():
                raise RuntimeError("Select a VESC CSV.")

            out_csv, out_png = make_output_paths(run_var.get(), outdir_var.get())

            process(
                force_audio_csv=force_var.get(),
                vesc_csv=vesc_var.get(),
                out_csv=out_csv,
                out_png=out_png,
                pole_pairs=float(pole_pairs_var.get()),
                merge_tolerance_s=float(merge_tol_var.get()),
                force_drop_frac=float(force_drop_frac_var.get()),
                audio_drop_db=float(audio_drop_db_var.get()),
                drop_sustain_s=float(drop_sustain_var.get()),
                search_last_frac=float(search_last_frac_var.get()),
            )

            messagebox.showinfo(
                "Done",
                f"Saved:\n{out_csv}\n{out_png}",
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Layout
    frm = ttk.Frame(root)
    frm.pack(fill="both", expand=True, **pad)

    # Inputs
    ttk.Label(frm, text="Force/Audio CSV:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=force_var, width=70).grid(row=0, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_force).grid(row=0, column=2)

    ttk.Label(frm, text="VESC CSV:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=vesc_var, width=70).grid(row=1, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_vesc).grid(row=1, column=2)

    ttk.Label(frm, text="Run name (prefix):").grid(row=2, column=0, sticky="w")
    ttk.Entry(frm, textvariable=run_var, width=30).grid(row=2, column=1, sticky="w")

    ttk.Label(frm, text="Output folder:").grid(row=3, column=0, sticky="w")
    ttk.Entry(frm, textvariable=outdir_var, width=70).grid(row=3, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_outdir).grid(row=3, column=2)

    # Params (compact)
    params = ttk.LabelFrame(frm, text="Parameters")
    params.grid(row=4, column=0, columnspan=3, sticky="we", **pad)

    def add_param(r, label, var):
        ttk.Label(params, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(params, textvariable=var, width=12).grid(row=r, column=1, sticky="w")

    add_param(0, "Pole pairs:", pole_pairs_var)
    add_param(0, "Merge tol (s):", merge_tol_var)
    add_param(1, "Force drop frac:", force_drop_frac_var)
    add_param(1, "Audio drop dB:", audio_drop_db_var)
    add_param(2, "Drop sustain (s):", drop_sustain_var)
    add_param(2, "Search last frac:", search_last_frac_var)

    # Run button
    ttk.Button(frm, text="Run", command=run).grid(row=5, column=0, columnspan=3, sticky="we", **pad)

    frm.columnconfigure(1, weight=1)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(
        description="Combine force/audio CSV with VESC CSV by aligning VESC last sample to force+audio drop-off."
    )

    # NEW: GUI mode
    ap.add_argument("--gui", action="store_true", help="Launch GUI instead of CLI mode")

    ap.add_argument("--force_audio_csv", help="Path to *_combined_event_aligned.csv")
    ap.add_argument("--vesc_csv", help="Path to VESC export CSV (semicolon-delimited)")

    # NEW: Run naming/output directory
    ap.add_argument("--run_name", default="run", help="Run name prefix for output files (date/time appended)")
    ap.add_argument("--out_dir", default=".", help="Output directory (used when out_csv/out_png not provided)")

    # Still allow manual overrides
    ap.add_argument("--out_csv", default=None, help="Output CSV filename (overrides --run_name/--out_dir)")
    ap.add_argument("--out_png", default=None, help="Output plot PNG filename (overrides --run_name/--out_dir)")

    ap.add_argument("--pole_pairs", type=float, default=7.0, help="Motor pole pairs (ERPM / pole_pairs = mechanical RPM)")
    ap.add_argument("--merge_tolerance_s", type=float, default=0.10, help="Max time difference allowed when joining (seconds)")

    ap.add_argument("--force_drop_frac", type=float, default=0.15, help="Force drop threshold as fraction of peak force")
    ap.add_argument("--audio_drop_db", type=float, default=8.0, help="Audio drop threshold: peak - this dBFS")
    ap.add_argument("--drop_sustain_s", type=float, default=0.5, help="Drop must hold for this long (seconds)")
    ap.add_argument("--search_last_frac", type=float, default=0.40, help="Search only in last fraction of force/audio window")

    args = ap.parse_args()

    if args.gui:
        launch_gui()
        return

    # CLI mode requires the two inputs
    if not args.force_audio_csv or not args.vesc_csv:
        raise SystemExit("CLI mode requires --force_audio_csv and --vesc_csv (or use --gui).")

    # NEW: determine outputs
    if args.out_csv and args.out_png:
        out_csv, out_png = args.out_csv, args.out_png
    else:
        out_csv, out_png = make_output_paths(args.run_name, args.out_dir)
        # If user provided only one override, respect it
        if args.out_csv:
            out_csv = args.out_csv
        if args.out_png:
            out_png = args.out_png

    process(
        force_audio_csv=args.force_audio_csv,
        vesc_csv=args.vesc_csv,
        out_csv=out_csv,
        out_png=out_png,
        pole_pairs=args.pole_pairs,
        merge_tolerance_s=args.merge_tolerance_s,
        force_drop_frac=args.force_drop_frac,
        audio_drop_db=args.audio_drop_db,
        drop_sustain_s=args.drop_sustain_s,
        search_last_frac=args.search_last_frac,
    )


if __name__ == "__main__":
    main()
