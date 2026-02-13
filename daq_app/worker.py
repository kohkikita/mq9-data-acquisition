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
    FORCE_THRESHOLD_N, PRE_SAMPLES, POST_SAMPLES, AUTO_STOP_SECONDS,
    START_ABOVE_CYCLES, END_BELOW_CYCLES,
    AUDIO_FS, AUDIO_CHANNELS, RMS_WINDOW_S,
    VESC_POLL_HZ, VESC_CMD_HZ
)
from .utils import find_stm32_port, parse_stm32_line
from .audio import AudioRecorder
from .vesc import VESCInterface, VESCConfig, VESCBackground
from .postprocess import (
    postprocess_event_aligned,
    save_force_audio_plot,
    get_event_intervals_from_raw_csv,
    write_event_only_wav,
    compute_audio_spectrogram
)


class RunPaths:
    # defined in gui.py; imported there; worker only relies on attributes
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

                # Start VESC background loop (this is what makes ramping smooth)
                vesc_stop = threading.Event()
                vesc_bg = VESCBackground(vesc, poll_hz=VESC_POLL_HZ, cmd_hz=VESC_CMD_HZ, stop_evt=vesc_stop)
                vesc_bg.start()
                self.gui_queue.put(("status", f"VESC background loop running (poll={VESC_POLL_HZ} Hz, cmd={VESC_CMD_HZ} Hz)"))

            # ---------------- Start run timebase + audio ----------------
            t0 = time.perf_counter()

            self.gui_queue.put(("status", "Starting audio (temporary full recording)..."))
            audio = AudioRecorder(self.paths.wav_full_path, AUDIO_FS, AUDIO_CHANNELS, device=self.mic_device)
            audio.start()

            # ---------------- Event detection state ----------------
            pre_buffer = deque(maxlen=PRE_SAMPLES)
            in_event = False
            post_remaining = 0
            event_id = 0
            saw_any_event = False
            below_since = None

            above_count = 0
            below_count = 0

            # ---------------- Main logging loop ----------------
            with open(self.paths.raw_event_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id", "pc_time_iso", "pc_elapsed_s", "stm32_time_ms", "force_N", "raw_line",
                    "vesc_rpm", "vesc_v_in_V", "vesc_i_motor_A", "vesc_i_in_A", "vesc_duty",
                    "vesc_temp_mos_C", "vesc_temp_motor_C",
                ])

                self.gui_queue.put(("status", f"Recording... ({self.paths.base_name})"))

                while not self._stop_req.is_set():
                    now_pc = time.perf_counter()

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
                    t_elapsed = time.perf_counter() - t0
                    now_iso = datetime.now().isoformat()

                    # Snapshot latest VESC telemetry (updated by background thread)
                    vesc_vals = vesc.snapshot() if vesc is not None else {
                        "vesc_rpm": np.nan,
                        "vesc_v_in_V": np.nan,
                        "vesc_i_motor_A": np.nan,
                        "vesc_i_in_A": np.nan,
                        "vesc_duty": np.nan,
                        "vesc_temp_mos_C": np.nan,
                        "vesc_temp_motor_C": np.nan,
                    }

                    self.gui_queue.put((
                        "line",
                        f"t={stm32_ms:8d} ms | pc={t_elapsed:8.3f} s | "
                        f"F={force_N:7.3f} N | "
                        f"VESC rpm={vesc_vals['vesc_rpm'] if np.isfinite(vesc_vals['vesc_rpm']) else 'NA'}"
                    ))

                    row = [
                        event_id, now_iso, t_elapsed, stm32_ms, force_N, line,
                        vesc_vals["vesc_rpm"],
                        vesc_vals["vesc_v_in_V"],
                        vesc_vals["vesc_i_motor_A"],
                        vesc_vals["vesc_i_in_A"],
                        vesc_vals["vesc_duty"],
                        vesc_vals["vesc_temp_mos_C"],
                        vesc_vals["vesc_temp_motor_C"],
                    ]

                    if not in_event:
                        pre_buffer.append(row)

                        if force_N >= FORCE_THRESHOLD_N:
                            above_count += 1
                            if above_count >= START_ABOVE_CYCLES:
                                in_event = True
                                post_remaining = 0
                                event_id += 1
                                saw_any_event = True
                                below_since = None
                                below_count = 0

                                self.gui_queue.put((
                                    "status",
                                    f"EVENT {event_id} START (above for {START_ABOVE_CYCLES} cycles)"
                                ))

                                # Flush pre-buffer into the new event
                                for r in pre_buffer:
                                    r[0] = event_id
                                    writer.writerow(r)
                                f.flush()
                                pre_buffer.clear()

                                row[0] = event_id
                                writer.writerow(row)
                                f.flush()
                        else:
                            above_count = 0

                            if saw_any_event:
                                if below_since is None:
                                    below_since = now_pc
                                elif (now_pc - below_since) >= AUTO_STOP_SECONDS:
                                    self.gui_queue.put((
                                        "status",
                                        f"AUTO-STOP: below {FORCE_THRESHOLD_N} N for {AUTO_STOP_SECONDS:.1f}s"
                                    ))
                                    break

                    else:
                        row[0] = event_id
                        writer.writerow(row)
                        f.flush()

                        if post_remaining == 0:
                            if force_N < FORCE_THRESHOLD_N:
                                below_count += 1
                                if below_count >= END_BELOW_CYCLES:
                                    post_remaining = POST_SAMPLES
                                    below_since = now_pc
                                    self.gui_queue.put((
                                        "status",
                                        f"EVENT {event_id} tail {POST_SAMPLES} samples (below for {END_BELOW_CYCLES} cycles)"
                                    ))
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

                        if force_N >= FORCE_THRESHOLD_N:
                            below_since = None
                            below_count = 0

            # ---------------- teardown: stop background + audio + serial ----------------
            self.gui_queue.put(("status", "Stopping audio..."))
            try:
                audio.stop()
            except Exception:
                pass

            try:
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
                RMS_WINDOW_S
            )

            self.gui_queue.put(("status", "Saving force/audio plot PNG..."))
            save_force_audio_plot(self.paths.combined_csv, self.paths.plot_png)

            self.gui_queue.put(("status", "Extracting event-only audio WAV..."))
            intervals = get_event_intervals_from_raw_csv(self.paths.raw_event_csv)
            if not intervals:
                raise RuntimeError("No events found in raw CSV, so no event-only audio to save.")
            write_event_only_wav(self.paths.wav_full_path, self.paths.wav_event_path, intervals)

            self.gui_queue.put(("status", "Computing spectrogram (event-only audio)..."))
            compute_audio_spectrogram(
                self.paths.wav_event_path,
                self.paths.spectrogram_csv,
                self.paths.spectrogram_png
            )

            try:
                os.remove(self.paths.wav_full_path)
            except Exception:
                pass

            self.gui_queue.put((
                "done",
                "Done.\n"
                f"Raw CSV: {self.paths.raw_event_csv}\n"
                f"WAV (event-only): {self.paths.wav_event_path}\n"
                f"Combined (force+vesc aligned): {self.paths.combined_csv}\n"
                f"Force/Audio Plot: {self.paths.plot_png}\n"
                f"Spectrogram CSV: {self.paths.spectrogram_csv}\n"
                f"Spectrogram Plot: {self.paths.spectrogram_png}\n"
            ))

        except Exception as e:
            # Ensure background thread is stopped on error too
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

            self.gui_queue.put(("error", str(e)))
