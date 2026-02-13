# daq_app/gui.py
from dataclasses import dataclass
import queue
import tkinter as tk
from tkinter import ttk, messagebox
import os

from .config import (
    RUNS_DIR, FORCE_THRESHOLD_N, START_ABOVE_CYCLES, END_BELOW_CYCLES,
    PRE_SAMPLES, POST_SAMPLES, AUTO_STOP_SECONDS, AUDIO_FS, AUDIO_CHANNELS, RMS_WINDOW_S,
    VESC_DEFAULT_ENABLED, VESC_DEFAULT_BAUD, VESC_MODES
)
from .utils import ensure_dir, now_stamp, sanitize_run_name
from .worker import RunWorker
from .vesc import VESCConfig


@dataclass
class RunPaths:
    base_name: str
    raw_event_csv: str          # TEMP file; deleted at end
    wav_full_path: str
    wav_event_path: str
    combined_csv: str
    spectrogram_csv: str
    spectrogram_png: str

    # New overlay plots
    overlay_all_png: str
    overlay_rpm_png: str
    overlay_power_png: str


def make_run_paths(run_name: str) -> RunPaths:
    ensure_dir(RUNS_DIR)
    stamp = now_stamp()
    run_name = sanitize_run_name(run_name)
    base = f"{run_name}_{stamp}"

    # NOTE: raw_event_csv is a TEMP file (deleted after post-processing)
    raw_csv = os.path.join(RUNS_DIR, f"{base}_TEMP_RAW.csv")

    wav_full = os.path.join(RUNS_DIR, f"{base}_FULL.wav")
    wav_event = os.path.join(RUNS_DIR, f"{base}.wav")
    combined = os.path.join(RUNS_DIR, f"{base}_combined_event_aligned.csv")

    spec_csv = os.path.join(RUNS_DIR, f"{base}_audio_spectrogram.csv")
    spec_png = os.path.join(RUNS_DIR, f"{base}_audio_spectrogram.png")

    overlay_all = os.path.join(RUNS_DIR, f"{base}_overlay_force_audio_rpm_power_duty.png")
    overlay_rpm = os.path.join(RUNS_DIR, f"{base}_overlay_force_audio_rpm.png")
    overlay_power = os.path.join(RUNS_DIR, f"{base}_overlay_force_audio_power.png")

    return RunPaths(
        base_name=base,
        raw_event_csv=raw_csv,
        wav_full_path=wav_full,
        wav_event_path=wav_event,
        combined_csv=combined,
        spectrogram_csv=spec_csv,
        spectrogram_png=spec_png,
        overlay_all_png=overlay_all,
        overlay_rpm_png=overlay_rpm,
        overlay_power_png=overlay_power,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STM32 Load Cell + USB Mic Logger (Event Audio Only) + VESC")
        self.geometry("1020x780")

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

        runfrm = ttk.Frame(frm)
        runfrm.pack(fill="x", pady=(8, 0))
        ttk.Label(runfrm, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(value="loadcell_run")
        ttk.Entry(runfrm, textvariable=self.run_name_var, width=32).pack(side="left", padx=(6, 0))

        # VESC UI
        vfrm = ttk.LabelFrame(frm, text="VESC (optional)", padding=8)
        vfrm.pack(fill="x", pady=(10, 0))

        self.vesc_enabled_var = tk.BooleanVar(value=VESC_DEFAULT_ENABLED)
        ttk.Checkbutton(vfrm, text="Enable VESC", variable=self.vesc_enabled_var).grid(row=0, column=0, sticky="w", padx=6, pady=2)

        ttk.Label(vfrm, text="Port (blank=auto):").grid(row=0, column=1, sticky="w", padx=6, pady=2)
        self.vesc_port_var = tk.StringVar(value="")
        ttk.Entry(vfrm, textvariable=self.vesc_port_var, width=14).grid(row=0, column=2, sticky="w", padx=6, pady=2)

        ttk.Label(vfrm, text="Baud:").grid(row=0, column=3, sticky="w", padx=6, pady=2)
        self.vesc_baud_var = tk.StringVar(value=str(VESC_DEFAULT_BAUD))
        ttk.Entry(vfrm, textvariable=self.vesc_baud_var, width=10).grid(row=0, column=4, sticky="w", padx=6, pady=2)

        ttk.Label(vfrm, text="Mode:").grid(row=1, column=1, sticky="w", padx=6, pady=2)
        self.vesc_mode_var = tk.StringVar(value="duty")  # default now duty
        ttk.Combobox(vfrm, textvariable=self.vesc_mode_var, values=VESC_MODES, width=12, state="readonly").grid(row=1, column=2, sticky="w", padx=6, pady=2)

        ttk.Label(vfrm, text="Setpoint:").grid(row=1, column=3, sticky="w", padx=6, pady=2)
        self.vesc_setpoint_var = tk.StringVar(value="1")  # default setpoint
        ttk.Entry(vfrm, textvariable=self.vesc_setpoint_var, width=10).grid(row=1, column=4, sticky="w", padx=6, pady=2)
        ttk.Label(vfrm, text="(rpm / A / duty 0-1)").grid(row=1, column=5, sticky="w", padx=6, pady=2)

        # --- Ramp controls ---
        ttk.Label(vfrm, text="Ramp RPM/s:").grid(row=2, column=1, sticky="w", padx=6, pady=2)
        self.vesc_ramp_rpm_var = tk.StringVar(value="3000")
        ttk.Entry(vfrm, textvariable=self.vesc_ramp_rpm_var, width=10).grid(row=2, column=2, sticky="w", padx=6, pady=2)

        ttk.Label(vfrm, text="Ramp duty/s:").grid(row=2, column=3, sticky="w", padx=6, pady=2)
        self.vesc_ramp_duty_var = tk.StringVar(value="0.10")  # default ramp duty/s
        ttk.Entry(vfrm, textvariable=self.vesc_ramp_duty_var, width=10).grid(row=2, column=4, sticky="w", padx=6, pady=2)

        self.vesc_ramp_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vfrm, text="Enable ramp", variable=self.vesc_ramp_enable_var).grid(
            row=2, column=5, sticky="w", padx=6, pady=2
        )

        self.vesc_hold_final_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vfrm, text="Hold final duty", variable=self.vesc_hold_final_var).grid(
            row=3, column=5, sticky="w", padx=6, pady=2
        )

        # settings summary
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
        ttk.Label(params, text=f"Output dir: {RUNS_DIR}/").grid(row=2, column=2, sticky="w", padx=6, pady=2)

        logfrm = ttk.LabelFrame(frm, text="Live Log", padding=8)
        logfrm.pack(fill="both", expand=True)

        self.text = tk.Text(logfrm, height=22, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(logfrm, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

    def _build_vesc_cfg(self) -> VESCConfig:
        enabled = bool(self.vesc_enabled_var.get())
        port = self.vesc_port_var.get().strip() or None

        try:
            baud = int(self.vesc_baud_var.get().strip())
        except Exception:
            baud = VESC_DEFAULT_BAUD

        mode = (self.vesc_mode_var.get() or "disabled").lower()
        if mode not in VESC_MODES:
            mode = "disabled"

        try:
            sp = float(self.vesc_setpoint_var.get().strip())
        except Exception:
            sp = 0.0

        ramp_enable = bool(self.vesc_ramp_enable_var.get())
        try:
            ramp_rpm_per_s = float(self.vesc_ramp_rpm_var.get().strip())
        except Exception:
            ramp_rpm_per_s = 3000.0

        try:
            ramp_duty_per_s = float(self.vesc_ramp_duty_var.get().strip())
        except Exception:
            ramp_duty_per_s = 0.10

        hold_final_duty = bool(self.vesc_hold_final_var.get())

        return VESCConfig(
            enabled=enabled,
            port=port,
            baud=baud,
            mode=mode,
            setpoint=sp,
            ramp_enable=ramp_enable,
            ramp_rpm_per_s=ramp_rpm_per_s,
            ramp_duty_per_s=ramp_duty_per_s,
            hold_final_duty=hold_final_duty,
        )

    def start_run(self):
        if self.worker is not None:
            return
        try:
            run_name = self.run_name_var.get()
            self.current_paths = make_run_paths(run_name)

            self.status_var.set(f"Preparing run: {self.current_paths.base_name}")
            self.text.insert("end", f"\n=== START RUN: {self.current_paths.base_name} ===\n")
            self.text.see("end")

            vcfg = self._build_vesc_cfg()
            self.worker = RunWorker(self.gui_queue, self.current_paths, mic_device=None, vesc_cfg=vcfg)
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
