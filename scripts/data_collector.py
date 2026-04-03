# data_collector.py
import serial
import numpy as np
import cv2
import os
import time
import struct
from pathlib import Path
from collections import defaultdict

# ── Configurazione ──────────────────────────────────────────────
PORT        = "COM3"          # Cambia con la tua porta (es. /dev/ttyUSB0)
BAUD_RATE   = 921600
IMG_W, IMG_H = 96, 96
FRAME_SIZE  = IMG_W * IMG_H * 3
DATASET_DIR = Path("dataset/raw")
CLASSES     = ["red", "green", "blue", "off"]
TARGET_PER_CLASS = 300

HEADER = bytes([0xAA, 0xBB])
FOOTER = bytes([0xCC, 0xDD])

# ── Setup directory ──────────────────────────────────────────────
for cls in CLASSES:
    (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)

# ── Contatori ────────────────────────────────────────────────────
counters = defaultdict(int)
for cls in CLASSES:
    counters[cls] = len(list((DATASET_DIR / cls).glob("*.png")))

KEY_MAP = {ord('r'): 'red', ord('g'): 'green',
           ord('b'): 'blue', ord('o'): 'off',
           ord('q'): None}

# ── Funzioni di protocollo ───────────────────────────────────────
def sync_to_header(ser: serial.Serial) -> bool:
    """Sincronizza lo stream sull'header 0xAA 0xBB."""
    buf = bytearray()
    deadline = time.time() + 3.0
    while time.time() < deadline:
        byte = ser.read(1)
        if not byte:
            continue
        buf.append(byte[0])
        if len(buf) >= 2 and buf[-2:] == HEADER:
            return True
    return False

def read_frame(ser: serial.Serial) -> np.ndarray | None:
    """Legge un frame completo secondo il protocollo custom."""
    if not sync_to_header(ser):
        print("[WARN] Timeout sincronizzazione header")
        return None

    size_bytes = ser.read(4)
    if len(size_bytes) < 4:
        return None
    payload_size = struct.unpack('>I', size_bytes)[0]

    if payload_size != FRAME_SIZE:
        print(f"[WARN] Size inattesa: {payload_size} (attesa {FRAME_SIZE})")
        ser.read(payload_size + 2)  # flush
        return None

    payload = ser.read(payload_size)
    footer  = ser.read(2)

    if footer != FOOTER:
        print("[WARN] Footer mancante, frame scartato")
        return None

    # Ricostruzione: RGB888 → BGR per OpenCV
    frame_rgb = np.frombuffer(payload, dtype=np.uint8).reshape((IMG_H, IMG_W, 3))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr

def save_frame(frame: np.ndarray, cls: str) -> str:
    ts = int(time.time() * 1000)
    filename = DATASET_DIR / cls / f"{cls}_{ts:013d}.png"
    cv2.imwrite(str(filename), frame)
    counters[cls] += 1
    return str(filename)

def draw_hud(frame: np.ndarray) -> np.ndarray:
    """Overlay con statistiche sul frame di preview."""
    display = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)
    display = cv2.copyMakeBorder(display, 0, 100, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))
    y = 400
    for i, cls in enumerate(CLASSES):
        count  = counters[cls]
        pct    = min(count / TARGET_PER_CLASS, 1.0)
        color  = (0, 200, 0) if pct >= 1.0 else (0, 165, 255)
        bar_w  = int(80 * pct)
        x_off  = 10 + i * 95
        cv2.rectangle(display, (x_off, y+5), (x_off+80, y+20), (80,80,80), -1)
        cv2.rectangle(display, (x_off, y+5), (x_off+bar_w, y+20), color, -1)
        label  = f"{cls[0].upper()}:{count}"
        cv2.putText(display, label, (x_off, y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    cv2.putText(display, "R=red G=green B=blue O=off Q=quit",
                (10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    return display

# ── Main loop ────────────────────────────────────────────────────
def main():
    print(f"Connessione a {PORT} @ {BAUD_RATE} baud...")
    with serial.Serial(PORT, BAUD_RATE, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        print("Connesso. Premi un tasto per salvare il frame corrente.")

        while True:
            frame = read_frame(ser)
            if frame is None:
                continue

            display = draw_hud(frame)
            cv2.imshow("ESP32-CAM — Data Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in KEY_MAP and KEY_MAP[key]:
                cls  = KEY_MAP[key]
                path = save_frame(frame, cls)
                print(f"[SAVE] {path}  |  {cls}: {counters[cls]}/{TARGET_PER_CLASS}")

        cv2.destroyAllWindows()
    print("\nRiepilogo finale:")
    for cls, cnt in counters.items():
        print(f"  {cls:6s}: {cnt:4d} immagini")

if __name__ == "__main__":
    main()