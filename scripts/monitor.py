"""
monitor.py — LED Classifier  |  Live Preview + Inference Results
═══════════════════════════════════════════════════════════════════════════════
Riceve dall'ESP32:
  1. Frame binario RGB888 96x96  →  visualizzato in finestra OpenCV
  2. JSON con classe e confidenza →  sovrapposto al frame come overlay

Protocollo atteso (identico al data acquisition streamer):
  [0xAA 0xBB] [size 4B BE] [payload RGB888] [0xCC 0xDD]
  {"class":"red","confidence":0.97,...}\n

Uso:
  python monitor.py
  python monitor.py --port COM12
  python monitor.py --port /dev/ttyUSB0 --log session.csv
"""

import serial
import struct
import json
import time
import argparse
import csv
import threading
import queue
from datetime import datetime
from collections import deque, defaultdict

import numpy as np
import cv2

# ── Configurazione ────────────────────────────────────────────────────────────
PORT        = "COM12"
BAUD_RATE   = 921600
IMG_W, IMG_H = 96, 96
FRAME_SIZE  = IMG_W * IMG_H * 2
HEADER      = bytes([0xAA, 0xBB])
FOOTER      = bytes([0xCC, 0xDD])

# Scala del preview (96x96 è piccolo — lo ingrandiamo)
PREVIEW_SCALE  = 5          # finestra: 480x480 pixel
PANEL_HEIGHT   = 180        # altezza pannello info sotto il frame
WINDOW_NAME    = "LED Classifier — Live Preview"

# Colori BGR per ogni classe
CLASS_COLORS = {
    "red"      : (60,  60,  220),
    "green"    : (60,  200, 60),
    "blue"     : (220, 100, 60),
    "no_led"   : (160, 160, 160),
    "uncertain": (0,   200, 220),
}

CLASSES = ["red", "green", "blue", "no_led"]


# ════════════════════════════════════════════════════════════════════════════
# Lettore seriale (thread separato)
# ════════════════════════════════════════════════════════════════════════════

def serial_reader(ser: serial.Serial,
                  frame_q: queue.Queue,
                  json_q:  queue.Queue,
                  stop_evt: threading.Event):
    """
    Legge continuamente dalla seriale.
    - Quando trova un header 0xAA 0xBB → legge un frame binario e lo mette in frame_q
    - Quando trova una riga JSON → la mette in json_q
    - Tutto il resto → stampato come log di boot/debug
    """
    buf = bytearray()

    while not stop_evt.is_set():
        # Leggi chunk disponibili
        waiting = ser.in_waiting
        if waiting == 0:
            time.sleep(0.001)
            continue

        chunk = ser.read(min(waiting, 4096))
        buf.extend(chunk)

        # Processa il buffer
        while len(buf) >= 2:

            # ── Cerca header binario ─────────────────────────────────────────
            idx = buf.find(HEADER)
            if idx == -1:
                # Nessun header: cerca righe JSON/testo nel buffer
                newline = buf.find(b'\n')
                if newline == -1:
                    # Niente di completo, aspetta altri dati
                    break
                line = buf[:newline].decode("utf-8", errors="ignore").strip()
                buf  = buf[newline+1:]
                _dispatch_text(line, json_q)
                continue

            # Testo/JSON prima dell'header binario
            if idx > 0:
                prefix = buf[:idx].decode("utf-8", errors="ignore")
                for line in prefix.split('\n'):
                    line = line.strip()
                    if line:
                        _dispatch_text(line, json_q)
                buf = buf[idx:]
                continue

            # ── Abbiamo l'header all'inizio: leggi frame completo ────────────
            # Serve: 2 (header) + 4 (size) + FRAME_SIZE (payload) + 2 (footer)
            needed = 2 + 4 + FRAME_SIZE + 2
            if len(buf) < needed:
                break   # non abbastanza dati, aspetta

            # Verifica size field
            size = struct.unpack('>I', buf[2:6])[0]
            if size != FRAME_SIZE:
                # CORREZIONE: Salta direttamente al prossimo header!
                next_header = buf.find(HEADER, 2)
                buf = buf[next_header:] if next_header != -1 else bytearray()
                continue

            # Verifica footer
            footer_start = 6 + FRAME_SIZE
            if buf[footer_start:footer_start+2] != FOOTER:
                # CORREZIONE: Salta direttamente al prossimo header!
                next_header = buf.find(HEADER, 2)
                buf = buf[next_header:] if next_header != -1 else bytearray()
                continue

            # Frame valido!
            raw_bytes = bytes(buf[6:6+FRAME_SIZE])
            buf = buf[needed:]

            # Ricostruisce immagine RGB → BGR per OpenCV
            raw_data = np.frombuffer(raw_bytes, dtype='>u2').reshape((IMG_H, IMG_W))
            r = ((raw_data >> 11) & 0x1F) << 3
            g = ((raw_data >> 5) & 0x3F) << 2
            b = (raw_data & 0x1F) << 3
            frame_bgr = np.stack([b, g, r], axis=-1).astype(np.uint8)

            # Non bloccare il thread se il consumer è lento: scartiamo frame vecchi
            if frame_q.full():
                try: frame_q.get_nowait()
                except queue.Empty: pass
            frame_q.put(frame_bgr)


def _dispatch_text(line: str, json_q: queue.Queue):
    """Smista una riga di testo: JSON → json_q, altrimenti stampa."""
    if line.startswith("{"):
        try:
            data = json.loads(line)
            if json_q.full():
                try: json_q.get_nowait()
                except queue.Empty: pass
            json_q.put(data)
        except json.JSONDecodeError:
            pass
    elif line:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  \033[90m[{ts}] {line}\033[0m")


# ════════════════════════════════════════════════════════════════════════════
# Rendering OpenCV
# ════════════════════════════════════════════════════════════════════════════

def build_display(frame_bgr: np.ndarray,
                  last_pred: dict,
                  history:   deque,
                  counters:  dict,
                  fps:       float) -> np.ndarray:
    """
    Costruisce il frame di visualizzazione completo:
      ┌─────────────────────────────┐
      │    Frame 480x480 (5x)       │
      │    + overlay classe         │
      ├─────────────────────────────┤
      │    Pannello info            │
      │    barre probabilità        │
      │    storico ultime N pred.   │
      └─────────────────────────────┘
    """
    W = IMG_W * PREVIEW_SCALE
    H = IMG_H * PREVIEW_SCALE

    # ── Ingrandisci frame con interpolazione nearest (mantiene pixel netti) ──
    big = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_NEAREST)

    cls        = last_pred.get("class", "uncertain")
    confidence = last_pred.get("confidence", 0.0)
    probs      = last_pred.get("probs", [0.0]*4)
    latency    = last_pred.get("ms", 0)
    color      = CLASS_COLORS.get(cls, (200, 200, 200))

    # ── Bordo colorato attorno al frame (indica la classe) ───────────────────
    border = 6
    big = cv2.copyMakeBorder(big, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=color)
    W2 = W + 2 * border
    H2 = H + 2 * border

    # ── Etichetta classe + confidenza (in alto a sinistra) ───────────────────
    label_text = f"{cls.upper()}  {confidence*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad = 6
    cv2.rectangle(big, (border, border),
                  (border + tw + pad*2, border + th + pad*2 + 4),
                  color, -1)
    cv2.putText(big, label_text,
                (border + pad, border + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── FPS + latenza (in alto a destra) ──────────────────────────────────────
    fps_text = f"{fps:.1f} FPS  {latency}ms"
    (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(big, fps_text,
                (W2 - fw - border - 4, border + fh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Pannello info ─────────────────────────────────────────────────────────
    panel = np.zeros((PANEL_HEIGHT, W2, 3), dtype=np.uint8)
    panel[:] = (28, 28, 40)

    # Barre di probabilità per ogni classe
    bar_x0  = 10
    bar_y0  = 18
    bar_w   = W2 - 20
    bar_h   = 18
    spacing = 36

    for i, (lbl, prob) in enumerate(zip(CLASSES, probs)):
        c   = CLASS_COLORS.get(lbl, (180, 180, 180))
        y   = bar_y0 + i * spacing
        filled = int(prob * bar_w)

        # Sfondo barra
        cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + bar_h),
                      (60, 60, 80), -1)
        # Fill
        if filled > 0:
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + filled, y + bar_h),
                          c, -1)
        # Etichetta
        pct_str = f"{lbl:<8} {prob*100:5.1f}%"
        text_col = (255, 255, 255) if lbl == cls else (180, 180, 180)
        cv2.putText(panel, pct_str,
                    (bar_x0 + 4, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 1)

        # Marcatore "◄" sulla classe vincente
        if lbl == cls:
            cv2.putText(panel, "<",
                        (bar_x0 + filled + 4, y + bar_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

    # Storico predizioni (ultimi N colori)
    hist_y = PANEL_HEIGHT - 22
    cv2.putText(panel, "Storico:", (bar_x0, hist_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 160), 1)
    sq = 14
    for j, past_cls in enumerate(list(history)[-30:]):
        hx = bar_x0 + 65 + j * (sq + 2)
        hc = CLASS_COLORS.get(past_cls, (120, 120, 120))
        cv2.rectangle(panel, (hx, hist_y - sq),
                      (hx + sq, hist_y), hc, -1)

    # ── Composizione finale ───────────────────────────────────────────────────
    return np.vstack([big, panel])


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  default=PORT)
    parser.add_argument("--baud",  default=BAUD_RATE, type=int)
    parser.add_argument("--log",   default=None, help="Salva predizioni in CSV")
    args = parser.parse_args()

    print(f"\n{'═'*55}")
    print(f"  LED Classifier — Live Monitor + Preview")
    print(f"  Porta: {args.port} @ {args.baud} baud")
    print(f"  Premi 'q' nella finestra OpenCV per uscire")
    print(f"{'═'*55}\n")

    # CSV logger opzionale
    csv_file = csv_writer = None
    if args.log:
        csv_file   = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp","class","confidence",
                              "p_red","p_green","p_blue","p_no_led","ms"])

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as e:
        print(f"  [ERRORE] Impossibile aprire {args.port}: {e}")
        return

    frame_q  = queue.Queue(maxsize=3)
    json_q   = queue.Queue(maxsize=10)
    stop_evt = threading.Event()

    reader_thread = threading.Thread(
        target=serial_reader,
        args=(ser, frame_q, json_q, stop_evt),
        daemon=True
    )
    reader_thread.start()
    print("  Thread seriale avviato. In attesa di dati dall'ESP32...\n")

    # Stato corrente
    last_pred  = {"class": "...", "confidence": 0.0,
                  "probs": [0.0]*4, "ms": 0}
    history    = deque(maxlen=30)
    counters   = defaultdict(int)

    # FPS calcolati sul preview
    fps_buf    = deque(maxlen=20)
    t_last     = time.time()

    # Frame placeholder (grigio) mostrato finché non arriva il primo frame
    placeholder = np.full((IMG_H, IMG_W, 3), 50, dtype=np.uint8)
    current_frame = placeholder.copy()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # ── Aggiorna frame ────────────────────────────────────────────────
            try:
                current_frame = frame_q.get_nowait()
                now  = time.time()
                fps_buf.append(1.0 / max(now - t_last, 1e-6))
                t_last = now
            except queue.Empty:
                pass

            # ── Aggiorna predizione ───────────────────────────────────────────
            try:
                while True:  # svuota la coda, usa solo l'ultima
                    data = json_q.get_nowait()
                    last_pred = data
                    cls = data.get("class", "uncertain")
                    history.append(cls)
                    counters[cls] += 1
                    if csv_writer:
                        p = data.get("probs", [0]*4)
                        csv_writer.writerow([
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            cls, f"{data.get('confidence',0):.4f}",
                            *[f"{x:.4f}" for x in p],
                            data.get("ms", 0)
                        ])
                        if csv_file: csv_file.flush()
            except queue.Empty:
                pass

            # ── Rendering ────────────────────────────────────────────────────
            fps = sum(fps_buf) / len(fps_buf) if fps_buf else 0.0
            display = build_display(current_frame, last_pred, history, counters, fps)
            cv2.imshow(WINDOW_NAME, display)

            # ── Gestione tasti ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:   # q o ESC per uscire
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        cv2.destroyAllWindows()
        ser.close()
        if csv_file: csv_file.close()

        total = sum(counters.values())
        if total:
            print(f"\n  Sessione terminata — {total} predizioni totali")
            for cls, cnt in sorted(counters.items(), key=lambda x: -x[1]):
                print(f"    {cls:<12} {cnt:4d}  ({cnt/total*100:5.1f}%)")
        if args.log:
            print(f"\n  Log salvato in: {args.log}")


if __name__ == "__main__":
    main()