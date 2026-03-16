# daq_app/worker.py
import os
import time
import csv
import queue
import threading
from collections import deque
from datetime import datetime

import numpy as np
import serial

from .config import (
    BAUD, SERIAL_TIMEOUT_S,
    AUDIO_FS, AUDIO_CHANNELS, RMS_WINDOW_S,
    VESC_POLL_HZ, VESC_CMD_HZ, FORCE_SCALE,
    RPM_PLATEAU_AUTOSTOP_ENABLE,
    RPM_PLATEAU_MIN_RPM,
    RPM_PLATEAU_EPS_RPM,
    RPM_PLATEAU_HOLD_S,
    RPM_PLATEAU_MIN_DUTY,
    RPM_PLATEAU_REQUIRE_VESC,
    RPM_PLATEAU_SMOOTH_SAMPLES,
)
from .utils import find_stm32_port, parse_stm32_line
from .audio import AudioRecorder
from .vesc import VESCInterface, VESCConfig, VESCBackground
from .postprocess import (
    postprocess_event_aligned,
    get_event_intervals_from_raw_csv,
    write_event_only_wav,
    compute_audio_spectrogram,
    save_overlay_force_audio_rpm_power_duty,
    save_overlay_force_audio_rpm,
    save_overlay_force_audio_power,
)


class RunPaths:
    # defined in gui.py; worker only relies on attributes
    pass


class RunWorker(threading.Thread):
    def __init__(self, gui_queue: queue.Queue, paths, mic_device=None, vesc_cfg: VESCConfig | None = None):
        super().__init__(daemon=True)
        self.gui_queue = gui_queue
        self.paths = paths
        self.mic_device = mic_device
        self.vesc_cfg = vesc_cfg
        self._stop_req = threading.Event()

    def request_stop(self):
        self._stop_req.set()

    def run(self):
        vesc = None
        vesc_bg = None
        vesc_stop = None
        audio = None
        ser = None

        try:
            # ---------------- STM32 connect ----------------
            port = find_stm32_port()
            self.gui_queue.put(("status", f"Serial: {port} @ {BAUD}"))

            ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT_S)
            time.sleep(2.0)

            # ---------------- Optional VESC connect ----------------
            if self.vesc_cfg and self.vesc_cfg.enabled and (self.vesc_cfg.mode or "disabled") != "disabled":
                self.gui_queue.put(("status", "Connecting to VESC..."))
                vesc = VESCInterface(self.vesc_cfg)
                vesc.open()
                self.gui_queue.put(("status", f"VESC connected ({vesc.ser.port} @ {self.vesc_cfg.baud})"))

                vesc_stop = threading.Event()
                vesc_bg = VESCBackground(vesc, poll_hz=VESC_POLL_HZ, cmd_hz=VESC_CMD_HZ, stop_evt=vesc_stop)
                vesc_bg.start()
                self.gui_queue.put(("status", f"VESC background loop running (poll={VESC_POLL_HZ} Hz, cmd={VESC_CMD_HZ} Hz)"))

            # ---------------- Start run timebase + audio ----------------
            t0 = time.perf_counter()

            self.gui_queue.put(("status", "Starting audio (temporary full recording)..."))
            audio = AudioRecorder(self.paths.wav_full_path, AUDIO_FS, AUDIO_CHANNELS, device=self.mic_device)
            audio.start()

            # Entire run is treated as one continuous event
            event_id = 1

            # RPM plateau auto-stop state
            rpm_peak = None
            rpm_peak_t = None
            plateau_armed = False
            rpm_hist = deque(maxlen=max(1, int(RPM_PLATEAU_SMOOTH_SAMPLES)))

            # ---------------- Main logging loop (TEMP raw CSV) ----------------
            with open(self.paths.raw_event_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id", "pc_time_iso", "pc_elapsed_s", "stm32_time_ms", "force_N", "raw_line",
                    "vesc_rpm", "vesc_v_in_V", "vesc_i_motor_A", "vesc_i_in_A", "vesc_duty",
                    "vesc_temp_mos_C", "vesc_power_W",
                ])

                self.gui_queue.put(("status", f"Recording continuously... ({self.paths.base_name})"))

                while not self._stop_req.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode(errors="ignore").strip()
                    if not line:
                        continue

                    parsed = parse_stm32_line(line)
                    if not parsed:
                        continue

                    force_N, stm32_ms = parsed
                    force_N = FORCE_SCALE * force_N
                    t_elapsed = time.perf_counter() - t0
                    now_iso = datetime.now().isoformat()

                    vesc_vals = vesc.snapshot() if vesc is not None else {
                        "vesc_rpm": np.nan,
                        "vesc_v_in_V": np.nan,
                        "vesc_i_motor_A": np.nan,
                        "vesc_i_in_A": np.nan,
                        "vesc_duty": np.nan,
                        "vesc_temp_mos_C": np.nan,
                        "vesc_power_W": np.nan,
                    }

                    duty_display = vesc_vals["vesc_duty"] if np.isfinite(vesc_vals["vesc_duty"]) else "NA"
                    rpm_display = vesc_vals["vesc_rpm"] if np.isfinite(vesc_vals["vesc_rpm"]) else "NA"

                    self.gui_queue.put((
                        "line",
                        f"t={stm32_ms:8d} ms | pc={t_elapsed:8.3f} s | "
                        f"F={force_N:7.3f} N | "
                        f"VESC Duty={duty_display} | "
                        f"VESC rpm={rpm_display}"
                    ))

                    writer.writerow([
                        event_id,
                        now_iso,
                        t_elapsed,
                        stm32_ms,
                        force_N,
                        line,
                        vesc_vals["vesc_rpm"],
                        vesc_vals["vesc_v_in_V"],
                        vesc_vals["vesc_i_motor_A"],
                        vesc_vals["vesc_i_in_A"],
                        vesc_vals["vesc_duty"],
                        vesc_vals["vesc_temp_mos_C"],
                        vesc_vals["vesc_power_W"],
                    ])
                    f.flush()

                    # ---------------- RPM plateau auto-stop ----------------
                    if RPM_PLATEAU_AUTOSTOP_ENABLE:
                        rpm_raw = float(vesc_vals["vesc_rpm"]) if np.isfinite(vesc_vals["vesc_rpm"]) else np.nan
                        duty = float(vesc_vals["vesc_duty"]) if np.isfinite(vesc_vals["vesc_duty"]) else np.nan

                        if np.isfinite(rpm_raw):
                            rpm_hist.append(rpm_raw)

                        rpm_eval = float(np.mean(rpm_hist)) if len(rpm_hist) > 0 else np.nan

                        vesc_ok = vesc is not None
                        duty_ok = np.isfinite(duty) and (abs(duty) >= RPM_PLATEAU_MIN_DUTY)
                        rpm_ok = np.isfinite(rpm_eval) and (abs(rpm_eval) >= RPM_PLATEAU_MIN_RPM)

                        if ((not RPM_PLATEAU_REQUIRE_VESC) or vesc_ok) and duty_ok and rpm_ok:
                            if not plateau_armed:
                                plateau_armed = True
                                rpm_peak = rpm_eval
                                rpm_peak_t = t_elapsed
                                self.gui_queue.put(("status", f"RPM plateau monitor armed at {rpm_eval:.0f} rpm"))
                            else:
                                if (rpm_peak is None) or (rpm_eval > (rpm_peak + RPM_PLATEAU_EPS_RPM)):
                                    rpm_peak = rpm_eval
                                    rpm_peak_t = t_elapsed
                                elif (rpm_peak_t is not None) and ((t_elapsed - rpm_peak_t) >= RPM_PLATEAU_HOLD_S):
                                    self.gui_queue.put((
                                        "status",
                                        f"AUTO-STOP: RPM plateau detected "
                                        f"(peak {rpm_peak:.0f} rpm, no new peak for {RPM_PLATEAU_HOLD_S:.1f}s)"
                                    ))
                                    break

            # ---------------- teardown: stop background + audio + serial ----------------
            self.gui_queue.put(("status", "Stopping audio..."))
            try:
                if audio is not None:
                    audio.stop()
            except Exception:
                pass

            try:
                if ser is not None:
                    ser.close()
            except Exception:
                pass

            if vesc_bg is not None and vesc_stop is not None:
                vesc_stop.set()
                vesc_bg.join(timeout=1.0)

            if vesc is not None:
                try:
                    vesc.close()
                except Exception:
                    pass

            # ---------------- post-processing ----------------
            self.gui_queue.put(("status", "Post-processing: creating combined CSV (aligned to force + VESC)..."))
            postprocess_event_aligned(
                self.paths.raw_event_csv,
                self.paths.wav_full_path,
                self.paths.combined_csv,
                RMS_WINDOW_S,
            )

            self.gui_queue.put(("status", "Saving run WAV..."))
            intervals = get_event_intervals_from_raw_csv(self.paths.raw_event_csv)
            if not intervals:
                raise RuntimeError("No logged samples found in raw CSV.")
            write_event_only_wav(self.paths.wav_full_path, self.paths.wav_event_path, intervals)

            self.gui_queue.put(("status", "Computing spectrogram..."))
            compute_audio_spectrogram(self.paths.wav_event_path, self.paths.spectrogram_csv, self.paths.spectrogram_png)

            self.gui_queue.put(("status", "Saving overlay plots..."))
            save_overlay_force_audio_rpm_power_duty(self.paths.combined_csv, self.paths.overlay_all_png)
            save_overlay_force_audio_rpm(self.paths.combined_csv, self.paths.overlay_rpm_png)
            save_overlay_force_audio_power(self.paths.combined_csv, self.paths.overlay_power_png)

            # Delete temp files user doesn't want saved
            try:
                os.remove(self.paths.wav_full_path)
            except Exception:
                pass
            try:
                os.remove(self.paths.raw_event_csv)
            except Exception:
                pass

            self.gui_queue.put((
                "done",
                "Done.\n"
                f"WAV: {self.paths.wav_event_path}\n"
                f"Combined (force+vesc aligned): {self.paths.combined_csv}\n"
                f"Spectrogram CSV: {self.paths.spectrogram_csv}\n"
                f"Spectrogram Plot: {self.paths.spectrogram_png}\n"
                f"Overlay (Force+Audio+RPM+Power+Duty): {self.paths.overlay_all_png}\n"
                f"Overlay (Force+Audio+RPM): {self.paths.overlay_rpm_png}\n"
                f"Overlay (Force+Audio+Power): {self.paths.overlay_power_png}\n"
            ))

        except Exception as e:
            try:
                if audio is not None:
                    audio.stop()
            except Exception:
                pass

            try:
                if ser is not None:
                    ser.close()
            except Exception:
                pass

            try:
                if vesc_bg is not None and vesc_stop is not None:
                    vesc_stop.set()
                    vesc_bg.join(timeout=1.0)
            except Exception:
                pass

            try:
                if vesc is not None:
                    vesc.close()
            except Exception:
                pass

            try:
                if hasattr(self.paths, "raw_event_csv") and self.paths.raw_event_csv and os.path.exists(self.paths.raw_event_csv):
                    os.remove(self.paths.raw_event_csv)
            except Exception:
                pass

            self.gui_queue.put(("error", str(e)))