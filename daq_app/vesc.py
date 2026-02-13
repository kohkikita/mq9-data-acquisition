# daq_app/vesc.py
"""
VESC UART/USB telemetry + command streaming with smooth ramping.

Provides:
  - VESCConfig: configuration including ramp rates
  - VESCInterface: open/close, request_values, poll (decode), snapshot, send_command (ramped)
  - VESCBackground: thread that runs request_values + send_command + poll at fixed rates
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass

import numpy as np
import serial
from serial.tools import list_ports


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


# Safety clamps (adjust for your setup)
VESC_MAX_RPM = 200000.0
VESC_MAX_DUTY = 0.95


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
    hold_final_duty: bool = True

    # Duty startup assist (helps sensorless starts in duty mode)
    duty_kick_enable: bool = True
    duty_kick_s: float = 0.20
    duty_kick_value: float = 0.10
    duty_min_start: float = 0.03


# ------------------------ VESC Interface ------------------------

class VESCInterface:
    """
    Minimal VESC helper using PyVESC.

      - request_values(): send telemetry request (prefers GetValuesSelect if available)
      - poll(): non-blocking RX decode; updates latest telemetry
      - send_command(): send control command; ramps rpm/duty using cfg ramp params
      - snapshot(): copy of latest telemetry dict
    """

    def __init__(self, cfg: VESCConfig):
        self.cfg = cfg
        self.ser: serial.Serial | None = None
        self.rxbuf = bytearray()
        self.lock = threading.Lock()

        # For duty kick window
        self._duty_kick_until: float | None = None

        # Telemetry
        self.latest = {
            "vesc_rpm": np.nan,
            "vesc_v_in_V": np.nan,
            "vesc_i_motor_A": np.nan,
            "vesc_i_in_A": np.nan,
            "vesc_duty": np.nan,
            "vesc_temp_mos_C": np.nan,
            "vesc_temp_motor_C": np.nan,
        }

        # Ramp state
        self._cmd_rpm = 0.0
        self._cmd_duty = 0.0
        self._last_cmd_t: float | None = None

        # Lazy import
        try:
            import pyvesc  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "VESC enabled, but PyVESC is not installed.\n"
                "Install with: pip install pyvesc-fix\n"
                f"Import error: {e}"
            )
        self.pyvesc = pyvesc

        # ✅ NEW: choose best telemetry request message
        # Prefer GetValuesSelect() if available in this PyVESC version.
        self._get_values_msg_factory = self._pick_get_values_factory()

    def _pick_get_values_factory(self):
        """
        Returns a callable that constructs the best telemetry request message.
        Prefer GetValuesSelect (if present), else fall back to GetValues.
        """
        # Some forks expose GetValuesSelect, some do not.
        if hasattr(self.pyvesc, "GetValuesSelect"):
            # Most PyVESC forks expect a mask/selection. Others default to "all".
            # We'll try calling it with no args first; if that errors, fall back to GetValues.
            def factory():
                try:
                    return self.pyvesc.GetValuesSelect()
                except TypeError:
                    # signature requires args in this fork -> fall back
                    return self.pyvesc.GetValues()
            return factory

        # Default classic request
        def factory():
            return self.pyvesc.GetValues()
        return factory

    def open(self):
        port = (self.cfg.port or "").strip()
        if not port:
            port = find_vesc_port()

        # NOTE: On USB CDC, baud is usually ignored by firmware/driver,
        # but Serial() still requires an int.
        self.ser = serial.Serial(port, int(self.cfg.baud), timeout=0.01)
        time.sleep(0.1)

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def _send_msg(self, msg):
        if self.ser is None:
            return
        pkt = self.pyvesc.encode(msg)
        self.ser.write(pkt)

    def request_values(self):
        """
        ✅ Updated: uses GetValuesSelect when available (more compatible on some firmware),
        otherwise falls back to GetValues.
        """
        msg = self._get_values_msg_factory()
        self._send_msg(msg)

    def poll(self):
        """
        Non-blocking read/decode. Updates self.latest when values decoded.
        """
        if self.ser is None:
            return

        n = self.ser.in_waiting
        if n:
            self.rxbuf += self.ser.read(n)

        while True:
            msg, consumed = self.pyvesc.decode(bytes(self.rxbuf))
            if consumed:
                self.rxbuf = self.rxbuf[consumed:]
            if not msg:
                break

            name = msg.__class__.__name__.lower()
            if name in ("getvalues", "getvaluesresponse", "values", "getvaluesselect", "getvaluesselectresponse"):
                d = {}
                for k in ("rpm", "v_in", "current_motor", "current_in", "duty_now", "temp_mos", "temp_motor"):
                    if hasattr(msg, k):
                        d[k] = getattr(msg, k)

                with self.lock:
                    self.latest["vesc_rpm"] = float(d.get("rpm", np.nan))
                    self.latest["vesc_v_in_V"] = float(d.get("v_in", np.nan))
                    self.latest["vesc_i_motor_A"] = float(d.get("current_motor", np.nan))
                    self.latest["vesc_i_in_A"] = float(d.get("current_in", np.nan))
                    self.latest["vesc_duty"] = float(d.get("duty_now", np.nan))
                    self.latest["vesc_temp_mos_C"] = float(d.get("temp_mos", np.nan))
                    self.latest["vesc_temp_motor_C"] = float(d.get("temp_motor", np.nan))

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.latest)

    def send_command(self):
        """
        Stream a control command based on cfg.mode and cfg.setpoint.

        Ramping:
          - rpm ramps at cfg.ramp_rpm_per_s
          - duty ramps at cfg.ramp_duty_per_s
          - current is immediate (no ramp)
        """
        mode = (self.cfg.mode or "disabled").lower()
        target = float(self.cfg.setpoint)

        now = time.perf_counter()
        if self._last_cmd_t is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_cmd_t)
        self._last_cmd_t = now

        ramp_en = bool(self.cfg.ramp_enable)

        if mode == "rpm":
            rpm_rate = float(self.cfg.ramp_rpm_per_s)
            target = clamp(target, -VESC_MAX_RPM, VESC_MAX_RPM)
            if ramp_en and dt > 0.0:
                self._cmd_rpm = ramp_toward(self._cmd_rpm, target, rpm_rate * dt)
            else:
                self._cmd_rpm = target
            self._send_msg(self.pyvesc.SetRPM(int(round(self._cmd_rpm))))
            return

        if mode == "duty":
            duty_rate = float(self.cfg.ramp_duty_per_s)
            hold_final = bool(self.cfg.hold_final_duty)

            target = clamp(target, -VESC_MAX_DUTY, VESC_MAX_DUTY)

            # ---- Kickstart logic (helps sensorless start in duty mode) ----
            kick_en = bool(getattr(self.cfg, "duty_kick_enable", False))
            kick_s = float(getattr(self.cfg, "duty_kick_s", 0.0))
            kick_val = float(getattr(self.cfg, "duty_kick_value", 0.0))
            min_start = float(getattr(self.cfg, "duty_min_start", 0.0))

            stopped = abs(self._cmd_duty) < 1e-4

            if kick_en and stopped and abs(target) > 1e-4:
                if self._duty_kick_until is None or now >= self._duty_kick_until:
                    self._duty_kick_until = now + max(0.0, kick_s)

            if self._duty_kick_until is not None and now < self._duty_kick_until:
                cmd = clamp(abs(kick_val), 0.0, VESC_MAX_DUTY) * (1.0 if target >= 0 else -1.0)
                self._cmd_duty = cmd
                self._send_msg(self.pyvesc.SetDutyCycle(int(round(self._cmd_duty * 100000.0))))
                return
            else:
                self._duty_kick_until = None

            # ---- Normal ramp/step ----
            if ramp_en:
                self._cmd_duty = ramp_toward(self._cmd_duty, target, duty_rate * dt)
                if abs(target) > 1e-4 and abs(self._cmd_duty) < abs(min_start):
                    self._cmd_duty = abs(min_start) * (1.0 if target >= 0 else -1.0)
            else:
                self._cmd_duty = target

            reached = abs(self._cmd_duty - target) < 1e-5
            if reached and (not hold_final):
                self._cmd_duty = 0.0
                self._send_msg(self.pyvesc.SetDutyCycle(0))
                return

            self._send_msg(self.pyvesc.SetDutyCycle(int(round(self._cmd_duty * 100000.0))))
            return

        if mode == "current":
            amps = float(target)
            self._send_msg(self.pyvesc.SetCurrent(int(round(amps * 1000.0))))
            return

        # disabled/no-op
        return


# ------------------------ Background thread ------------------------

class VESCBackground(threading.Thread):
    """
    Runs VESC request_values + send_command + poll at fixed rates,
    independent of the STM32 serial read loop (smooth ramping).
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
