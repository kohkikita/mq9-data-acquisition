# daq_app/vesc.py
"""
VESC UART/USB telemetry + command streaming with smooth ramping.

Telemetry:
  - Uses native VESC packet framing + CRC16 + COMM_GET_VALUES decode
  - Logs:
      vesc_rpm            (scaled by /7)
      vesc_v_in_V
      vesc_i_motor_A
      vesc_i_in_A
      vesc_duty
      vesc_temp_mos_C
      vesc_power_W        (v_in * i_in)

Commands:
  - COMM_SET_DUTY / COMM_SET_RPM / COMM_SET_CURRENT
  - Duty uses ramp state machine (ramp up -> hold -> ramp down -> disarm)

Note:
  - Motor temp is intentionally NOT tracked/logged anymore.
"""

from __future__ import annotations

import struct
import time
import threading
from dataclasses import dataclass

import numpy as np
import serial
from serial.tools import list_ports
from .config import *


# ------------------------ COMM IDs ------------------------
COMM_GET_VALUES = 4
COMM_SET_DUTY = 5
COMM_SET_CURRENT = 6
COMM_SET_RPM = 8


# ------------------------ Safety clamps ------------------------
VESC_MAX_RPM = 200000.0
VESC_MAX_DUTY = 1.0


# ------------------------ Helpers ------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ramp_toward(current: float, target: float, max_step: float) -> float:
    """
    Move current toward target by at most max_step (>=0).
    If max_step <= 0, hold current (do NOT jump to target).
    """
    if max_step <= 0.0:
        return current
    if target > current:
        return min(current + max_step, target)
    return max(current - max_step, target)


def find_vesc_port() -> str:
    """
    Best-effort VESC serial port selection.
    Many VESCs show up as generic USB-serial, so this is heuristic-only.
    """
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found (cannot find VESC).")

    prefer = ["vesc", "bldc", "chibios", "cp210", "silicon labs", "ftdi", "usb serial"]
    for p in ports:
        desc = (p.description or "").lower()
        manu = (p.manufacturer or "").lower()
        if any(k in desc for k in prefer) or any(k in manu for k in prefer):
            return p.device

    if len(ports) == 1:
        return ports[0].device

    lines = ["Could not uniquely identify VESC serial port.", "Available ports:"]
    for p in ports:
        lines.append(f"  {p.device}: {p.description} ({p.manufacturer})")
    raise RuntimeError("\n".join(lines))


# ------------------------ VESC packet framing + CRC16 ------------------------

def crc16_ccitt(data: bytes) -> int:
    """
    CRC16-CCITT (poly 0x1021), init 0. Standard VESC UART CRC.
    """
    crc = 0
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def vesc_pack(payload: bytes) -> bytes:
    """
    VESC frame format:
      [2][len][payload][crc_hi][crc_lo][3]            if len < 256
      [3][len_hi][len_lo][payload][crc_hi][crc_lo][3] if len >= 256
    """
    l = len(payload)
    crc = crc16_ccitt(payload)
    if l < 256:
        return bytes([2, l]) + payload + struct.pack(">H", crc) + bytes([3])
    return bytes([3]) + struct.pack(">H", l) + payload + struct.pack(">H", crc) + bytes([3])


def vesc_try_unpack(rxbuf: bytearray) -> list[bytes]:
    """
    Extract as many valid payloads as possible from rxbuf, removing consumed bytes.
    Robust against garbage/partial data.
    """
    out: list[bytes] = []

    while True:
        if len(rxbuf) < 6:
            break

        # Find start byte 2 or 3
        start_i = None
        for i, b in enumerate(rxbuf):
            if b in (2, 3):
                start_i = i
                break
        if start_i is None:
            rxbuf.clear()
            break
        if start_i > 0:
            del rxbuf[:start_i]

        if len(rxbuf) < 6:
            break

        start = rxbuf[0]

        # short frame
        if start == 2:
            l = rxbuf[1]
            frame_len = 2 + l + 2 + 1
            if len(rxbuf) < frame_len:
                break
            if rxbuf[frame_len - 1] != 3:
                del rxbuf[0:1]
                continue
            payload = bytes(rxbuf[2:2 + l])
            crc_rx = struct.unpack(">H", bytes(rxbuf[2 + l:2 + l + 2]))[0]
            if crc16_ccitt(payload) != crc_rx:
                del rxbuf[0:1]
                continue
            out.append(payload)
            del rxbuf[:frame_len]
            continue

        # long frame
        if start == 3:
            if len(rxbuf) < 7:
                break
            l = struct.unpack(">H", bytes(rxbuf[1:3]))[0]
            frame_len = 3 + l + 2 + 1
            if len(rxbuf) < frame_len:
                break
            if rxbuf[frame_len - 1] != 3:
                del rxbuf[0:1]
                continue
            payload = bytes(rxbuf[3:3 + l])
            crc_rx = struct.unpack(">H", bytes(rxbuf[3 + l:3 + l + 2]))[0]
            if crc16_ccitt(payload) != crc_rx:
                del rxbuf[0:1]
                continue
            out.append(payload)
            del rxbuf[:frame_len]
            continue

        del rxbuf[0:1]

    return out


def _get_i16_be(b: bytes, idx: int) -> tuple[int, int]:
    return struct.unpack(">h", b[idx:idx + 2])[0], idx + 2


def _get_i32_be(b: bytes, idx: int) -> tuple[int, int]:
    return struct.unpack(">i", b[idx:idx + 4])[0], idx + 4


def _get_scaled_i16(b: bytes, idx: int, scale: float) -> tuple[float, int]:
    v, idx = _get_i16_be(b, idx)
    return float(v) / scale, idx


def _get_scaled_i32(b: bytes, idx: int, scale: float) -> tuple[float, int]:
    v, idx = _get_i32_be(b, idx)
    return float(v) / scale, idx


# ------------------------ Config ------------------------

@dataclass
class VESCConfig:
    enabled: bool
    port: str | None
    baud: int
    mode: str
    setpoint: float

    ramp_enable: bool = True
    ramp_rpm_per_s: float = 3000.0
    ramp_duty_per_s: float = 0.10
    ramp_down: float = 0.5

    # If True: once ramp reaches duty setpoint, hold indefinitely.
    # If False: hold for hold_seconds then ramp down to 0 and DISARM.
    hold_final_duty: bool = True
    hold_seconds = VESC_DEFAULT_HOLD_TIME

    # Sensorless startup assist (duty mode)
    duty_kick_enable: bool = True
    duty_kick_s: float = 0.20
    duty_kick_value: float = 0.10
    duty_min_start: float = 0.03


# ------------------------ VESC Interface ------------------------

class VESCInterface:
    def __init__(self, cfg: VESCConfig):
        self.cfg = cfg
        self.ser: serial.Serial | None = None
        self.rxbuf = bytearray()

        self.lock = threading.Lock()
        self.latest = {
            "vesc_rpm": np.nan,         # NOTE: stored as RPM/7
            "vesc_v_in_V": np.nan,
            "vesc_i_motor_A": np.nan,
            "vesc_i_in_A": np.nan,
            "vesc_duty": np.nan,
            "vesc_temp_mos_C": np.nan,
            "vesc_power_W": np.nan,     # v_in * i_in
        }

        # --- command ramp state ---
        self._last_cmd_t: float | None = None
        self._cmd_rpm = 0.0
        self._cmd_duty = 0.0

        # duty state machine:
        #   idle -> ramp_up -> (hold_final OR hold_timed) -> ramp_down -> done(disarmed)
        self._duty_state: str = "idle"
        self._duty_target_latched: float = 0.0
        self._duty_hold_until: float | None = None
        self._duty_kick_until: float | None = None

    def open(self):
        port = (self.cfg.port or "").strip()
        if not port:
            port = find_vesc_port()
        self.ser = serial.Serial(port, int(self.cfg.baud), timeout=0.01)
        time.sleep(0.1)
        self._last_cmd_t = None

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def _write_payload(self, payload: bytes):
        if self.ser is None:
            return
        self.ser.write(vesc_pack(payload))

    # -------- Telemetry --------

    def request_values(self):
        self._write_payload(bytes([COMM_GET_VALUES]))

    def poll(self):
        if self.ser is None:
            return

        n = self.ser.in_waiting
        if n:
            self.rxbuf += self.ser.read(n)

        payloads = vesc_try_unpack(self.rxbuf)
        for p in payloads:
            if not p:
                continue
            if p[0] == COMM_GET_VALUES:
                try:
                    self._decode_get_values(p)
                except Exception:
                    pass

    def _decode_get_values(self, payload: bytes):
        """
        payload[0] == COMM_GET_VALUES
        Common field order:
          temp_fet (i16 / 10)
          temp_motor (i16 / 10)   [parsed but NOT stored]
          current_motor (i32 / 100)
          current_in    (i32 / 100)
          id            (i32 / 100)  [ignored]
          iq            (i32 / 100)  [ignored]
          duty_now      (i16 / 1000)
          rpm           (i32 / 1)
          v_in          (i16 / 10)
        """
        if len(payload) < 1 + 2 + 2 + 4 + 4 + 4 + 4 + 2 + 4 + 2:
            raise ValueError("GetValues payload too short")

        idx = 1
        temp_fet, idx = _get_scaled_i16(payload, idx, 10.0)
        temp_motor, idx = _get_scaled_i16(payload, idx, 10.0)   # parsed but unused
        i_motor, idx = _get_scaled_i32(payload, idx, 100.0)
        i_in, idx = _get_scaled_i32(payload, idx, 100.0)
        _, idx = _get_scaled_i32(payload, idx, 100.0)  # id (ignored)
        _, idx = _get_scaled_i32(payload, idx, 100.0)  # iq (ignored)
        duty, idx = _get_scaled_i16(payload, idx, 1000.0)
        rpm_raw, idx = _get_scaled_i32(payload, idx, 1.0)
        v_in, idx = _get_scaled_i16(payload, idx, 10.0)

        rpm_scaled = float(rpm_raw) / 7.0
        power_W = int(v_in) * float(i_in)

        with self.lock:
            self.latest["vesc_temp_mos_C"] = float(temp_fet)
            self.latest["vesc_i_motor_A"] = float(i_motor)
            self.latest["vesc_i_in_A"] = float(i_in)
            self.latest["vesc_duty"] = float(duty)
            self.latest["vesc_rpm"] = float(rpm_scaled)
            self.latest["vesc_v_in_V"] = float(v_in)
            self.latest["vesc_power_W"] = (power_W)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.latest)

    # -------- Command streaming / ramping --------

    def _dt(self) -> float:
        now = time.perf_counter()
        if self._last_cmd_t is None:
            self._last_cmd_t = now
            return 0.0
        dt = max(0.0, now - self._last_cmd_t)
        self._last_cmd_t = now
        return dt

    def _duty_set_state_for_new_target(self, target: float):
        self._duty_target_latched = target
        self._duty_hold_until = None
        self._duty_kick_until = None

        if abs(target) <= 1e-6:
            self._duty_state = "idle"
        else:
            self._duty_state = "ramp_up"

    def send_command(self):
        mode = (self.cfg.mode or "disabled").lower()
        target = float(self.cfg.setpoint)
        dt = self._dt()
        now = time.perf_counter()

        ramp_en = bool(self.cfg.ramp_enable)

        # RPM
        if mode == "rpm":
            target = clamp(target, -VESC_MAX_RPM, VESC_MAX_RPM)
            if ramp_en and dt > 0.0:
                self._cmd_rpm = ramp_toward(self._cmd_rpm, target, float(self.cfg.ramp_rpm_per_s) * dt)
            else:
                self._cmd_rpm = target
            val = int(round(self._cmd_rpm))
            payload = bytes([COMM_SET_RPM]) + struct.pack(">i", val)
            self._write_payload(payload)
            return

        # Current
        if mode == "current":
            amps = target
            val = int(round(amps * 1000.0))
            payload = bytes([COMM_SET_CURRENT]) + struct.pack(">i", val)
            self._write_payload(payload)
            return

        # Duty
        if mode != "duty":
            return

        target = clamp(target, -VESC_MAX_DUTY, VESC_MAX_DUTY)

        if abs(target - self._duty_target_latched) > 1e-6:
            self._duty_set_state_for_new_target(target)

        if self._duty_state == "done":
            self._cmd_duty = 0.0
            payload = bytes([COMM_SET_DUTY]) + struct.pack(">i", 0)
            self._write_payload(payload)
            return

        duty_rate = float(self.cfg.ramp_duty_per_s)
        hold_final = bool(self.cfg.hold_final_duty)
        hold_s = max(0.0, float(self.cfg.hold_seconds))

        kick_en = bool(getattr(self.cfg, "duty_kick_enable", False))
        kick_s = max(0.0, float(getattr(self.cfg, "duty_kick_s", 0.0)))
        kick_val = float(getattr(self.cfg, "duty_kick_value", 0.0))
        min_start = max(0.0, float(getattr(self.cfg, "duty_min_start", 0.0)))

        stopped = abs(self._cmd_duty) < 1e-4
        want_move = abs(target) > 1e-4

        if kick_en and stopped and want_move and self._duty_state in ("ramp_up", "idle"):
            if self._duty_kick_until is None or now >= self._duty_kick_until:
                self._duty_kick_until = now + kick_s

        if self._duty_kick_until is not None and now < self._duty_kick_until:
            cmd = clamp(abs(kick_val), 0.0, VESC_MAX_DUTY) * (1.0 if target >= 0 else -1.0)
            self._cmd_duty = cmd
            duty_int = int(round(self._cmd_duty * 100000.0))
            payload = bytes([COMM_SET_DUTY]) + struct.pack(">i", duty_int)
            self._write_payload(payload)
            return
        else:
            self._duty_kick_until = None

        if self._duty_state == "idle":
            self._cmd_duty = 0.0

        elif self._duty_state == "ramp_up":
            if ramp_en and dt > 0.0:
                self._cmd_duty = ramp_toward(self._cmd_duty, target, duty_rate * dt)
                if want_move and abs(self._cmd_duty) < min_start:
                    self._cmd_duty = min_start * (1.0 if target >= 0 else -1.0)
            else:
                self._cmd_duty = target

            reached = abs(self._cmd_duty - target) <= 1e-5
            if reached:
                if hold_final:
                    self._duty_state = "hold_final"
                else:
                    self._duty_state = "hold_timed"
                    self._duty_hold_until = now + hold_s

        elif self._duty_state == "hold_final":
            self._cmd_duty = target

        elif self._duty_state == "hold_timed":
            self._cmd_duty = target
            if (self._duty_hold_until is not None) and (now >= self._duty_hold_until):
                self._duty_state = "ramp_down"

        elif self._duty_state == "ramp_down":
            if ramp_en and dt > 0.0:
                self._cmd_duty = ramp_toward(self._cmd_duty, 0.0, self.ramp_down * dt)
            else:
                self._cmd_duty = 0.0

            if abs(self._cmd_duty) <= 1e-5:
                self._cmd_duty = 0.0
                self._duty_state = "done"

        else:
            self._cmd_duty = 0.0
            self._duty_state = "done"

        duty_int = int(round(self._cmd_duty * 100000.0))
        payload = bytes([COMM_SET_DUTY]) + struct.pack(">i", duty_int)
        self._write_payload(payload)


# ------------------------ Background thread ------------------------

class VESCBackground(threading.Thread):
    """
    Runs request_values + send_command + poll at fixed rates.
    """
    def __init__(self, vesc: VESCInterface, poll_hz: float, cmd_hz: float, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.vesc = vesc
        self.poll_dt = 1.0 / max(1.0, float(poll_hz))
        self.cmd_dt = 1.0 / max(1.0, float(cmd_hz))
        self.stop_evt = stop_evt

    def run(self):
        next_poll = time.perf_counter()
        next_cmd = time.perf_counter()

        while not self.stop_evt.is_set():
            now = time.perf_counter()

            if now >= next_poll:
                try:
                    self.vesc.request_values()
                except Exception:
                    pass
                next_poll = now + self.poll_dt

            if now >= next_cmd:
                try:
                    self.vesc.send_command()
                except Exception:
                    pass
                next_cmd = now + self.cmd_dt

            try:
                self.vesc.poll()
            except Exception:
                pass

            time.sleep(0.002)
