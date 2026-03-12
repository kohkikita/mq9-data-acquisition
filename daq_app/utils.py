# daq_app/utils.py
import os
import re
from datetime import datetime
from serial.tools import list_ports

LINE_RE = re.compile(r"Load=([+-]?\d+(?:\.\d+)?)\s*N,\s*t=(\d+)\s*ms")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sanitize_run_name(name: str) -> str:
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

    for p in ports:
        desc = (p.description or "").lower()
        manu = (p.manufacturer or "").lower()
        if any(k in desc for k in ["stm", "stlink", "nucleo", "stm32"]) or any(
            k in manu for k in ["stmicroelectronics", "st"]
        ):
            return p.device

    if len(ports) == 1:
        return ports[0].device

    lines = ["Could not uniquely identify STM32 serial port.", "Available ports:"]
    for p in ports:
        lines.append(f"  {p.device}: {p.description} ({p.manufacturer})")
    raise RuntimeError("\n".join(lines))


def parse_stm32_line(line: str):
    m = LINE_RE.search(line)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


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
