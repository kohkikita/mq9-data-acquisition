# daq_app/gui.py
from collections import deque
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass
import queue
import tkinter as tk
from tkinter import ttk, messagebox
import os

from .config import *
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
    VESC_FIELD_WIDTH = 14
    VESC_COMBO_WIDTH = 12

    def __init__(self):
        super().__init__()
        self.title("STM32 Load Cell + USB Mic Logger (Event Audio Only) + VESC")
        self.geometry("1020x780")
        self.minsize(980, 720)

        self.gui_queue = queue.Queue()
        self.worker = None
        self.current_paths = None

        self.gui_queue = queue.Queue()
        self.worker = None
        self.current_paths = None
        self.live_window_s = 10.0
        self.live_t = deque(maxlen=5000)
        self.live_rpm = deque(maxlen=5000)
        self.live_power = deque(maxlen=5000)
        self.settings_side_by_side = None
        self.vesc_compact_layout = None
        self.params_columns = None
        self._configure_theme()
        self._build_ui()
        self.bind("<Configure>", self._on_resize)
        self.after(50, self._poll_queue)
        self.after(100, self._update_plot)

    def _configure_theme(self):
        self.colors = {
            "bg": "#08090b",
            "panel": "#121418",
            "panel_alt": "#171a20",
            "border": "#2c3038",
            "text": "#f4f4f5",
            "muted": "#a4a7ae",
            "red": "#c41230",
            "red_hover": "#e51b3f",
            "red_dark": "#7a0c1d",
            "entry": "#0d0f13",
            "plot": "#0b0d10",
            "grid": "#343944",
        }

        self.configure(bg=self.colors["bg"])

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"])

        style.configure(
            "TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
        )
        style.configure(
            "Muted.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
        )
        style.configure(
            "Status.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Segoe UI Semibold", 10),
            padding=(10, 7),
        )

        style.configure(
            "TLabelframe",
            background=self.colors["bg"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=self.colors["bg"],
            foreground=self.colors["red"],
            font=("Segoe UI Semibold", 10),
        )

        style.configure(
            "TButton",
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            focusthickness=0,
            padding=(12, 7),
        )
        style.map(
            "TButton",
            background=[("active", self.colors["border"]), ("disabled", "#202329")],
            foreground=[("disabled", "#74777f")],
        )
        style.configure(
            "Accent.TButton",
            background=self.colors["red"],
            foreground="#ffffff",
            bordercolor=self.colors["red"],
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["red_hover"]), ("disabled", "#44101a")],
            foreground=[("disabled", "#9a8086")],
        )

        style.configure(
            "TCheckbutton",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            focuscolor=self.colors["bg"],
        )
        style.map(
            "TCheckbutton",
            background=[("active", self.colors["bg"])],
            foreground=[("active", self.colors["text"])],
        )

        style.configure(
            "TEntry",
            fieldbackground=self.colors["entry"],
            foreground=self.colors["text"],
            insertcolor=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=4,
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["entry"],
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            arrowcolor=self.colors["red"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.colors["entry"])],
            foreground=[("readonly", self.colors["text"])],
            selectbackground=[("readonly", self.colors["red_dark"])],
            selectforeground=[("readonly", "#ffffff")],
        )

    def _build_ui(self):
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm, style="Panel.TFrame", padding=10)
        top.pack(fill="x", pady=(0, 10))

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(side="left", fill="x", expand=True)

        self.start_btn = ttk.Button(top, text="Start Run", command=self.start_run, style="Accent.TButton")
        self.start_btn.pack(side="right", padx=(5, 0))

        self.stop_btn = ttk.Button(top, text="Stop Run", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="right", padx=(5, 0))

        runfrm = ttk.Frame(frm, padding=(0, 2, 0, 2))
        runfrm.pack(fill="x", pady=(0, 10))
        ttk.Label(runfrm, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(value="loadcell_run")
        ttk.Entry(runfrm, textvariable=self.run_name_var, width=32).pack(side="left", padx=(6, 0))

        self.settings_container = ttk.Frame(frm)
        self.settings_container.pack(fill="x", expand=True, pady=(0, 10))

        # VESC UI
        self.vesc_frame = ttk.LabelFrame(self.settings_container, text="VESC (optional)", padding=12)
        vfrm = self.vesc_frame

        self.vesc_enabled_var = tk.BooleanVar(value=VESC_DEFAULT_ENABLED)
        self.vesc_enable_check = ttk.Checkbutton(vfrm, text="Enable VESC", variable=self.vesc_enabled_var)

        self.vesc_port_label = ttk.Label(vfrm, text="Port (blank=auto):")
        self.vesc_port_var = tk.StringVar(value="")
        self.vesc_port_entry = ttk.Entry(vfrm, textvariable=self.vesc_port_var, width=self.VESC_FIELD_WIDTH)

        self.vesc_baud_label = ttk.Label(vfrm, text="Baud:")
        self.vesc_baud_var = tk.StringVar(value=str(VESC_DEFAULT_BAUD))
        self.vesc_baud_entry = ttk.Entry(vfrm, textvariable=self.vesc_baud_var, width=self.VESC_FIELD_WIDTH)

        self.vesc_mode_label = ttk.Label(vfrm, text="Mode:")
        self.vesc_mode_var = tk.StringVar(value=VESC_DEFAULT_MODE)  # default now duty
        self.vesc_mode_combo = ttk.Combobox(vfrm, textvariable=self.vesc_mode_var, values=VESC_MODES, width=self.VESC_COMBO_WIDTH, state="readonly")

        self.vesc_setpoint_label = ttk.Label(vfrm, text="Setpoint:")
        self.vesc_setpoint_var = tk.StringVar(value=VESC_DEFAULT_SETPOINT)  # default setpoint
        self.vesc_setpoint_entry = ttk.Entry(vfrm, textvariable=self.vesc_setpoint_var, width=self.VESC_FIELD_WIDTH)
        self.vesc_units_label = ttk.Label(vfrm, text="(rpm / A / duty 0-1)")

        # --- Ramp controls ---
        self.vesc_ramp_rpm_label = ttk.Label(vfrm, text="Ramp RPM/s:")
        self.vesc_ramp_rpm_var = tk.StringVar(value=VESC_DEFAULT_RAMP_RPM_PER_S)
        self.vesc_ramp_rpm_entry = ttk.Entry(vfrm, textvariable=self.vesc_ramp_rpm_var, width=self.VESC_FIELD_WIDTH)

        self.vesc_ramp_duty_label = ttk.Label(vfrm, text="Ramp duty/s:")
        self.vesc_ramp_duty_var = tk.StringVar(value=VESC_DEFAULT_RAMP_DUTY_PER_S)  # default ramp duty/s
        self.vesc_ramp_duty_entry = ttk.Entry(vfrm, textvariable=self.vesc_ramp_duty_var, width=self.VESC_FIELD_WIDTH)

        self.vesc_ramp_enable_var = tk.BooleanVar(value=VESC_DEFAULT_RAMP_ENABLE)
        self.vesc_ramp_enable_check = ttk.Checkbutton(vfrm, text="Enable ramp", variable=self.vesc_ramp_enable_var)

        self.vesc_hold_final_var = tk.BooleanVar(value=VESC_DEFAULT_HOLD_FINAL)
        self.vesc_hold_final_check = ttk.Checkbutton(vfrm, text="Hold final duty", variable=self.vesc_hold_final_var)

        # settings summary
        self.params_frame = ttk.LabelFrame(self.settings_container, text="Run Settings (edit in script constants)", padding=12)
        params = self.params_frame

        param_texts = (
            f"Threshold: {FORCE_THRESHOLD_N} N",
            f"Start debounce: {START_ABOVE_CYCLES} cycles >= threshold",
            f"End debounce: {END_BELOW_CYCLES} cycles < threshold",
            f"Pre: {PRE_SAMPLES} samples",
            f"Post: {POST_SAMPLES} samples",
            f"Auto-stop: {AUTO_STOP_SECONDS} s below",
            f"Audio: {AUDIO_FS} Hz, {AUDIO_CHANNELS} ch",
            f"RMS window: {RMS_WINDOW_S} s",
            f"Output dir: {RUNS_DIR}/",
        )
        self.param_labels = [ttk.Label(params, text=text) for text in param_texts]
        self._layout_settings()
        
        # NEW: live plot
        plotfrm = ttk.LabelFrame(frm, text="Live Plot", padding=12)
        plotfrm.pack(fill="both", expand=False, pady=(0, 10))

        self.fig = Figure(figsize=(10, 4.8), dpi=100)
        self.fig.patch.set_facecolor(self.colors["panel"])
        self.ax_rpm = self.fig.add_subplot(111)
        self.ax_power = self.ax_rpm.twinx()

        self.ax_rpm.set_facecolor(self.colors["plot"])
        self.ax_power.set_facecolor(self.colors["plot"])
        self.ax_rpm.set_title("Live RPM / Power")
        self.ax_rpm.set_xlabel("Time (s)")
        self.ax_rpm.set_ylabel("RPM")
        self.ax_power.set_ylabel("Power (W)")
        self.ax_rpm.grid(True, color=self.colors["grid"], alpha=0.55)

        logo_path = os.path.join(os.path.dirname(__file__), "assets", "propops_dark.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(__file__), "assets", "propops.png")
        if os.path.exists(logo_path):
            try:
                watermark = mpimg.imread(logo_path)
                logo_box = OffsetImage(
                    watermark,
                    zoom=0.22,
                    alpha=0.11,
                )
                self.ax_rpm.add_artist(
                    AnnotationBbox(
                        logo_box,
                        (0.5, 0.5),
                        xycoords=self.ax_rpm.transAxes,
                        frameon=False,
                        pad=0,
                        box_alignment=(0.5, 0.5),
                        zorder=0,
                    )
                )
            except Exception:
                pass

        for ax in (self.ax_rpm, self.ax_power):
            ax.tick_params(colors=self.colors["muted"])
            ax.xaxis.label.set_color(self.colors["muted"])
            ax.yaxis.label.set_color(self.colors["muted"])
            ax.title.set_color(self.colors["text"])
            for spine in ax.spines.values():
                spine.set_color(self.colors["border"])

        (self.rpm_line,) = self.ax_rpm.plot([], [], label="RPM", color="#4da3ff", linewidth=1.9)
        (self.power_line,) = self.ax_power.plot([], [], label="Power (W)", color=self.colors["red_hover"], linewidth=1.9)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plotfrm)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        logfrm = ttk.LabelFrame(frm, text="Live Log", padding=12)
        logfrm.pack(fill="x", expand=False)

        self.text = tk.Text(
            logfrm,
            height=6,
            wrap="none",
            bg=self.colors["entry"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground=self.colors["red_dark"],
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
            font=("Consolas", 9),
        )
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(logfrm, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

    def _on_resize(self, event):
        if event.widget is self:
            self._layout_settings()

    def _layout_settings(self):
        if not hasattr(self, "settings_container"):
            return

        width = self.winfo_width()
        side_by_side = width >= 1500
        vesc_compact = width < 1250
        if side_by_side:
            param_columns = 2 if width >= 1750 else 1
        else:
            param_columns = 1 if width < 760 else 2 if width < 1250 else 3

        if (
            self.settings_side_by_side == side_by_side
            and self.vesc_compact_layout == vesc_compact
            and self.params_columns == param_columns
        ):
            return

        self.settings_side_by_side = side_by_side
        self.vesc_compact_layout = vesc_compact
        self.params_columns = param_columns

        self._layout_vesc_fields(vesc_compact)
        self._layout_param_fields(param_columns)

        self.vesc_frame.grid_forget()
        self.params_frame.grid_forget()

        for col in range(2):
            self.settings_container.columnconfigure(col, weight=0)
        self.settings_container.rowconfigure(0, weight=0)
        self.settings_container.rowconfigure(1, weight=0)

        if side_by_side:
            self.settings_container.columnconfigure(0, weight=3, uniform="settings")
            self.settings_container.columnconfigure(1, weight=2, uniform="settings")
            self.vesc_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
            self.params_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        else:
            self.settings_container.columnconfigure(0, weight=1)
            self.vesc_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
            self.params_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=0)

    def _layout_vesc_fields(self, compact: bool):
        widgets = (
            self.vesc_enable_check,
            self.vesc_port_label,
            self.vesc_port_entry,
            self.vesc_baud_label,
            self.vesc_baud_entry,
            self.vesc_mode_label,
            self.vesc_mode_combo,
            self.vesc_setpoint_label,
            self.vesc_setpoint_entry,
            self.vesc_units_label,
            self.vesc_ramp_rpm_label,
            self.vesc_ramp_rpm_entry,
            self.vesc_ramp_duty_label,
            self.vesc_ramp_duty_entry,
            self.vesc_ramp_enable_check,
            self.vesc_hold_final_check,
        )
        for widget in widgets:
            widget.grid_forget()

        for col in range(6):
            self.vesc_frame.columnconfigure(col, weight=0)

        if compact:
            self.vesc_setpoint_label.configure(text="Setpoint (rpm / A / duty 0-1):")
            self.vesc_frame.columnconfigure(0, weight=0)
            self.vesc_frame.columnconfigure(1, weight=0)
            rows = (
                (self.vesc_port_label, self.vesc_port_entry),
                (self.vesc_baud_label, self.vesc_baud_entry),
                (self.vesc_mode_label, self.vesc_mode_combo),
                (self.vesc_setpoint_label, self.vesc_setpoint_entry),
                (self.vesc_ramp_rpm_label, self.vesc_ramp_rpm_entry),
                (self.vesc_ramp_duty_label, self.vesc_ramp_duty_entry),
            )

            self.vesc_enable_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            for row, (label, field) in enumerate(rows, start=1):
                label.grid(row=row, column=0, sticky="w", padx=6, pady=2)
                field.grid(row=row, column=1, sticky="w", padx=6, pady=2)

            self.vesc_ramp_enable_check.grid(row=7, column=1, sticky="w", padx=6, pady=2)
            self.vesc_hold_final_check.grid(row=8, column=1, sticky="w", padx=6, pady=2)
            return

        self.vesc_setpoint_label.configure(text="Setpoint:")
        self.vesc_enable_check.grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.vesc_port_label.grid(row=0, column=1, sticky="w", padx=6, pady=2)
        self.vesc_port_entry.grid(row=0, column=2, sticky="w", padx=6, pady=2)
        self.vesc_baud_label.grid(row=0, column=3, sticky="w", padx=6, pady=2)
        self.vesc_baud_entry.grid(row=0, column=4, sticky="w", padx=6, pady=2)
        self.vesc_mode_label.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        self.vesc_mode_combo.grid(row=1, column=2, sticky="w", padx=6, pady=2)
        self.vesc_setpoint_label.grid(row=1, column=3, sticky="w", padx=6, pady=2)
        self.vesc_setpoint_entry.grid(row=1, column=4, sticky="w", padx=6, pady=2)
        self.vesc_units_label.grid(row=1, column=5, sticky="w", padx=6, pady=2)
        self.vesc_ramp_rpm_label.grid(row=2, column=1, sticky="w", padx=6, pady=2)
        self.vesc_ramp_rpm_entry.grid(row=2, column=2, sticky="w", padx=6, pady=2)
        self.vesc_ramp_duty_label.grid(row=2, column=3, sticky="w", padx=6, pady=2)
        self.vesc_ramp_duty_entry.grid(row=2, column=4, sticky="w", padx=6, pady=2)
        self.vesc_ramp_enable_check.grid(row=2, column=5, sticky="w", padx=6, pady=2)
        self.vesc_hold_final_check.grid(row=3, column=5, sticky="w", padx=6, pady=2)

    def _layout_param_fields(self, columns: int):
        for label in self.param_labels:
            label.grid_forget()

        for col in range(3):
            self.params_frame.columnconfigure(col, weight=0)

        for index, label in enumerate(self.param_labels):
            row = index // columns
            col = index % columns
            self.params_frame.columnconfigure(col, weight=1)
            label.grid(row=row, column=col, sticky="w", padx=6, pady=2)

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
            # NEW: clear live data buffers and plot
            self.live_t.clear()
            self.live_rpm.clear()
            self.live_power.clear()

            self.rpm_line.set_data([], [])
            self.power_line.set_data([], [])

            self.ax_rpm.relim()
            self.ax_rpm.autoscale_view()
            self.ax_power.relim()
            self.ax_power.autoscale_view()

            self.canvas.draw_idle()
            
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
        
    # NEW: live plot update
    def _update_plot(self):
        try:
            if self.live_t:
                t_vals = list(self.live_t)
                rpm_vals = list(self.live_rpm)
                power_vals = list(self.live_power)

                t_max = t_vals[-1]
                t_min = max(0.0, t_max - self.live_window_s)

                idx0 = 0
                for i, t in enumerate(t_vals):
                    if t >= t_min:
                        idx0 = i
                        break

                x = t_vals[idx0:]
                y_rpm = rpm_vals[idx0:]
                y_power = power_vals[idx0:]

                self.rpm_line.set_data(x, y_rpm)
                self.power_line.set_data(x, y_power)

                self.ax_rpm.set_xlim(t_min, max(t_min + 0.1, t_max))

                rpm_clean = [v for v in y_rpm if v == v]
                if rpm_clean:
                    rmin = min(rpm_clean)
                    rmax = max(rpm_clean)
                    pad = max(100.0, (rmax - rmin) * 0.1 if rmax != rmin else 100.0)
                    self.ax_rpm.set_ylim(rmin - pad, rmax + pad)

                power_clean = [v for v in y_power if v == v]
                if power_clean:
                    pmin = min(power_clean)
                    pmax = max(power_clean)
                    pad = max(1.0, (pmax - pmin) * 0.1 if pmax != pmin else 1.0)
                    self.ax_power.set_ylim(pmin - pad, pmax + pad)

                self.canvas.draw_idle()

        finally:
            self.after(100, self._update_plot)
        
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
                
                # NEW: live sample update
                elif msg_type == "sample":
                    t = payload.get("t")
                    rpm = payload.get("rpm")
                    power = payload.get("power")

                    if t is not None:
                        self.live_t.append(float(t))
                        self.live_rpm.append(float(rpm) if rpm is not None else float("nan"))
                        self.live_power.append(float(power) if power is not None else float("nan"))

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
