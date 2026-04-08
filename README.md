# ESP32-CAM TinyML LED Classifier 🔴🟢🔵

## Descrizione Progetto

Un sistema di classificazione in tempo reale che riconosce i LED accesi (rosso, verde, blu) utilizzando una **ESP32-CAM** e un modello **TensorFlow Lite** ottimizzato per microcontrollori. Il progetto combina **machine learning embedded** con l'elaborazione di immagini per identificare lo stato dei LED con alta accuratezza.

### Funzionalità Principali
- ✅ Acquisizione immagini in tempo reale dalla fotocamera OV2640 integrata
- ✅ Classificazione di 4 classi: LED Rosso, Verde, Blu, Spento (No LED)
- ✅ Modello TensorFlow Lite compatto ed efficiente (RAM < 1MB)
- ✅ Streaming seriale a 921600 baud per acquisizione dati
- ✅ Soglia di confidenza configurabile (default 70%)
- ✅ Risoluzione 96x96px in RGB565 per ottimizzazione RAM

---

## Architettura del Progetto

```
esp32cam-tinyml-led-classifier/
├── src/
│   ├── main.cpp                 ← Firmware principale ESP32
│   └── led_model_data.h         ← Modello TFLite embedato
├── include/
│   ├── config.h                 ← Configurazioni PIN fotocamera
│   └── README
├── scripts/
│   ├── data_collector.py        ← Raccogliere immagini da ESP32
│   ├── augment_dataset.py       ← Aumentare il dataset (4x immagini)
│   ├── train.ipynb              ← Addestramento modello TFLite
│   ├── monitor.py               ← Monitoraggio seriale
│   └── serial_streamer.py       ← Streaming dati raw
├── dataset/
│   ├── raw/                     ← Immagini originali raccolte
│   │   ├── red/
│   │   ├── green/
│   │   ├── blue/
│   │   └── no_led/
│   └── augmented/               ← Dataset aumentato e statistiche
│       ├── dataset_stats.json
│       ├── red/
│       ├── green/
│       ├── blue/
│       └── no_led/
├── platformio.ini               ← Configurazione PlatformIO
├── requirements.txt             ← Dipendenze Python
└── README.md                    ← Questo file
```

---

## Hardware Richiesto

| Componente | Specifica |
|-----------|-----------|
| **Microcontrollore** | ESP32 (con PSRAM consigliata) |
| **Fotocamera** | OV2640 (Freenove WROVER) |
| **LED Indicatori** | Rosso, Verde, Blu (5mm, 5V) |
| **Connessione** | USB-UART (TTL 3.3V) |

### Pin Fotocamera Freenove WROVER
```
XCLK  → GPIO 21
SIOD  → GPIO 26  (SDA)
SIOC  → GPIO 27  (SCL)
Y2-Y9 → GPIO 4, 5, 18, 19, 36, 39, 34, 35
VSYNC → GPIO 25
HREF  → GPIO 23
PCLK  → GPIO 22
```

---

## Setup Iniziale

### 1. Preparazione Ambiente Python

```bash
# Creare virtual environment
python -m venv venv_tinyml

# Attivare (Windows)
venv_tinyml\Scripts\activate

# Attivare (Linux/Mac)
source venv_tinyml/bin/activate

# Installare dipendenze
pip install -r requirements.txt
```

### 2. Build e Flash del Firmware

```bash
# Utilizzare PlatformIO CLI
platformio run -e esp32cam -t upload

# Oppure da VS Code: PlatformIO > Upload
```

### 3. Configurare la Porta Seriale

Modificare `scripts/data_collector.py`:
```python
PORT = "COM12"  # Windows: COM3, COM12, etc.
                # Linux: /dev/ttyUSB0
BAUD_RATE = 921600
```

---

## Flusso di Lavoro - Addestramento Modello

### Fase 1: Raccolta Dati

```bash
python scripts/data_collector.py
```

**Comandi interattivi:**
- `r` → Salva frame come **RED**
- `g` → Salva frame come **GREEN**
- `b` → Salva frame come **BLUE**
- `n` → Salva frame come **NO_LED**
- `q` → Esci

**Obiettivo:** 300 immagini per classe (1200 totali)

```
dataset/raw/
├── red/      [300 immagini]
├── green/    [300 immagini]
├── blue/     [300 immagini]
└── no_led/   [300 immagini]
```

### Fase 2: Augmentazione Dataset

```bash
python scripts/augment_dataset.py
```

- Applica 4x rotazioni, flips e distorsioni geometriche
- **Preserva colore** (hue/saturation invariate)
- Output: ~1200 immagini per classe
- Genera `dataset_stats.json`

```
dataset/augmented/
├── red/              [~1200 immagini]
├── green/            [~1200 immagini]
├── blue/             [~1200 immagini]
├── no_led/           [~1200 immagini]
└── dataset_stats.json
```

### Fase 3: Addestramento Modello

Aprire e eseguire `scripts/train.ipynb`:
1. Carica dataset aumentato
2. Divide train/validation (80/20)
3. Addestra MobileNetV2 con transfer learning
4. Quantizza il modello (INT8)
5. Esporta come TFLite (`.tflite`)
6. Genera `led_model_data.h` embedato

### Fase 4: Deploy su ESP32

1. Rimpiazzare `src/led_model_data.h` con il nuovo modello
2. Ricompilare: `platformio run -e esp32cam -t upload`
3. Testare tramite monitor seriale

---

## Utilizzo del Firmware

### Monitoraggio Seriale

```bash
# Terminal integrale con debug
platformio device monitor -e esp32cam --speed 460800

# Oppure da VS Code: Monitor
```

**Output esempio:**
```
[INFO] Camera initialized
[INFO] TFLite interpreter created, Arena: 512KB
Frame 0001: RED (confidence: 0.96)
Frame 0002: RED (confidence: 0.94)
Frame 0003: NO_LED (confidence: 0.88)
...
```

### Streaming Dati Raw

```bash
python scripts/serial_streamer.py
```

Cattura il flusso di frame RGB565 per debug offline.

---

## Configurazioni Modificabili

### `src/main.cpp`

```cpp
#define CONFIDENCE_THRESHOLD  0.70f     // Soglia minima di confidenza
#define IMG_WIDTH             96        // Risoluzione
#define IMG_HEIGHT            96
#define IMG_CHANNELS          3         // RGB (3 canali)
#define TFLITE_ARENA_SIZE     (512 * 1024)  // Buffer per TFLite
#define STREAM_EVERY_N_FRAMES 1         // Invia 1 frame ogni N
```

### `platformio.ini`

```ini
monitor_speed = 460800         # Debug UART
upload_speed = 921600          # Upload speed
board_build.partitions = huge_app.csv  # Partizione grande
```

---

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| **Kernel Panic su init camera** | PSRAM cache fix attivo in platformio.ini |
| **OOM (Out of Memory)** | Aumentare PSRAM, ridurre arena TFLite o risoluzioni |
| **Bassa accuratezza** | Raccogliere più dati, migliorare illuminazione |
| **Disconnessione seriale** | Verificare cavo USB, ridurre baud rate |
| **Modello troppo grande** | Applicare quantizzazione INT8 o model pruning |

---

## Specifiche Tecniche

### Modello
- **Framework:** TensorFlow Lite Micro
- **Quantization:** INT8 per ESP32 (riduzione 4x)
- **Input:** Immagine 96×96×3 (RGB565)
- **Output:** 4 probabilità (softmax)
- **Latenza:** ~200-300ms per frame (ESP32-S3: ~50-100ms)

### Memoria
- **Flash:** ~1-2MB (dipende da quantizzazione)
- **RAM Statica:** ~512KB per arena TFLite
- **PSRAM Dinamica:** ~1-2MB per framebuffer

### Protocollo Seriale
```
[0xAA 0xBB] [96×96×2 bytes] [0xCC 0xDD]  ← Frame RGB565
|  HEADER  |  IMMAGINE     | FOOTER     |
```

---

## Dipendenze

### C/C++ (PlatformIO)
- `espressif32` (ESP32 core)
- `esp32-camera` (Driver OV2640)
- `TensorFlowLite_ESP32` v1.0.0

### Python
```
tensorflow~=2.14
keras~=2.14
opencv-python~=4.8
numpy~=1.24
pillow~=10.0
matplotlib~=3.8
pyserial~=3.5
```

---

## Best Practices

✅ **Raccolta Dati**
- Variare illuminazione (naturale, artificiale, mista)
- Includere diverse angolazioni
- Catturare transizioni LED on/off
- Balancio classi ~300 immagini ciascuna

✅ **Augmentazione**
- Preservare colore (il discriminatore principale)
- 4-8x multiplier consigliato
- Validare con `dataset_stats.json`

✅ **Addestramento**
- Usare validation set per overfitting detection
- Early stopping dopo ~20 epoche
- Quantizzare DOPO aver raggiunto accuratezza target

✅ **Deployment**
- Testare in esercizio prima di deployare
- Monitorare confidenza (refrain sotto 60%)
- Cherry-pick wrong predictions per retraining

---

## Sviluppi Futuri

- [ ] Aggiungere classificazione dell'intensità LED (0-100%)
- [ ] Supporto per più colori LED (rosso+verde = giallo?)
- [ ] Riconoscimento pattern temporali (blinking detection)
- [ ] Inferenza su ESP32-S3 (dual-core, più veloce)
- [ ] Web dashboard per monitoring remoto
- [ ] OTA (Over-The-Air) updates

---

## Risorse

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-CAM Documentation](https://docs.espressif.com/projects/esp-idf/)
- [PlatformIO Documentation](https://docs.platformio.org/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---

## Licenza

Open Source - Libero per uso educativo e commerciale

---

## Autore

**Matteo Donofrio** - matteodonofrio003

---

**Ultimo aggiornamento:** Aprile 2026  
**Status:** In sviluppo attivo ✓
