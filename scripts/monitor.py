"""
monitor.py
══════════════════════════════════════════════════════════════════
Monitor seriale per LED Classifier ESP32-CAM
Legge l'output JSON del firmware e lo visualizza in modo leggibile.

Uso:
    python monitor.py
    python monitor.py --port COM12 --verbose
    python monitor.py --port /dev/ttyUSB0 --log results.csv
"""

import serial
import json
import argparse
import csv
import time
from datetime import datetime
from collections import deque, defaultdict

# Mappa colori ANSI per terminale
ANSI = {
    "red"     : "\033[91m",
    "green"   : "\033[92m",
    "blue"    : "\033[94m",
    "no_led"  : "\033[90m",
    "uncertain": "\033[93m",
    "reset"   : "\033[0m",
    "bold"    : "\033[1m",
}

# Finestra scorrevole per statistiche (ultimi N frame)
STATS_WINDOW = 50


def colorize(label: str, text: str) -> str:
    color = ANSI.get(label, "")
    return f"{color}{text}{ANSI['reset']}"


def draw_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM LED Classifier Monitor")
    parser.add_argument("--port",    default="COM3",    help="Porta seriale")
    parser.add_argument("--baud",    default=115200,    type=int)
    parser.add_argument("--verbose", action="store_true", help="Mostra distribuzione completa")
    parser.add_argument("--log",     default=None,      help="Salva risultati in CSV")
    args = parser.parse_args()

    print(f"\n{'═'*55}")
    print(f"  LED Classifier Monitor")
    print(f"  Porta: {args.port} @ {args.baud} baud")
    print(f"  Premi Ctrl+C per uscire")
    print(f"{'═'*55}\n")

    # Setup CSV logger opzionale
    csv_file   = None
    csv_writer = None
    if args.log:
        csv_file   = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "class", "confidence",
                              "p_red", "p_green", "p_blue", "p_no_led", "ms"])
        print(f"  Log CSV: {args.log}\n")

    # Statistiche
    class_counts  = defaultdict(int)
    latency_buf   = deque(maxlen=STATS_WINDOW)
    total_frames  = 0

    try:
        with serial.Serial(args.port, args.baud, timeout=2) as ser:
            ser.reset_input_buffer()
            print("  Connesso. In attesa di dati...\n")

            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Ignora righe di boot/log non-JSON
                if not line.startswith("{"):
                    print(f"  \033[90m[ESP32] {line}\033[0m")
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cls         = data.get("class", "?")
                confidence  = data.get("confidence", 0.0)
                probs       = data.get("probs", [0, 0, 0, 0])
                latency_ms  = data.get("ms", 0)
                timestamp   = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                total_frames += 1
                class_counts[cls] += 1
                latency_buf.append(latency_ms)

                # ── Visualizzazione principale ────────────────────────────────
                conf_bar = draw_bar(confidence)
                cls_str  = colorize(cls, f"{cls:<10}")
                conf_str = colorize(cls, f"{confidence*100:5.1f}%")

                print(f"  [{timestamp}]  {cls_str}  {conf_bar}  {conf_str}  {latency_ms:3d}ms")

                if args.verbose:
                    labels = ["red", "green", "blue", "no_led"]
                    for i, (lbl, p) in enumerate(zip(labels, probs)):
                        bar    = draw_bar(p, 15)
                        marker = " ◄" if lbl == cls else "  "
                        print(f"             {lbl:<8} {colorize(lbl, bar)}  {p*100:5.1f}%{marker}")
                    print()

                # ── Statistiche ogni 25 frame ─────────────────────────────────
                if total_frames % 25 == 0:
                    avg_lat = sum(latency_buf) / len(latency_buf)
                    fps     = 1000.0 / avg_lat if avg_lat > 0 else 0
                    print(f"\n  ── Stats (ultimi {total_frames} frame) ──")
                    for lbl in ["red", "green", "blue", "no_led", "uncertain"]:
                        cnt = class_counts[lbl]
                        if cnt > 0:
                            pct = cnt / total_frames * 100
                            print(f"     {colorize(lbl, lbl):<20} {cnt:4d}  ({pct:5.1f}%)")
                    print(f"     Latenza media      : {avg_lat:.1f} ms  (~{fps:.1f} FPS)")
                    print()

                # ── Log CSV ───────────────────────────────────────────────────
                if csv_writer:
                    csv_writer.writerow([
                        timestamp, cls, f"{confidence:.4f}",
                        *[f"{p:.4f}" for p in probs],
                        latency_ms
                    ])
                    csv_file.flush()

    except KeyboardInterrupt:
        print(f"\n\n  Interrotto dopo {total_frames} frame.")
        if total_frames > 0:
            print(f"\n  Distribuzione finale:")
            for lbl, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                pct = cnt / total_frames * 100
                print(f"    {colorize(lbl, lbl):<20} {cnt:4d}  ({pct:5.1f}%)")

    except serial.SerialException as e:
        print(f"\n  [ERRORE] {e}")
        print(f"  Verifica che la porta '{args.port}' sia corretta e libera.")

    finally:
        if csv_file:
            csv_file.close()
            print(f"\n  Log salvato in: {args.log}")


if __name__ == "__main__":
    main()