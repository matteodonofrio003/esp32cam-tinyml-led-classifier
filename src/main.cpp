/**
 * main.cpp
 * ════════════════════════════════════════════════════════════════════════════
 * LED Classifier — Inferenza on-device con TFLite Micro
 * Hardware : ESP32-CAM Freenove WROVER
 * Modello  : led_cnn INT8 (led_model_data.h)
 * Classi   : 0=red | 1=green | 2=blue | 3=no_led
 *
 * Flusso principale:
 *   setup() → inizializza camera + TFLite interpreter
 *   loop()  → cattura frame → preprocess → inferenza → pubblica risultato
 *
 * Output seriale (115200 baud):
 *   JSON per ogni frame  {"class":"red","confidence":0.97,"ms":112}
 *   Oppure testo leggibile se VERBOSE_OUTPUT = true
 * ════════════════════════════════════════════════════════════════════════════
 */

#include <Arduino.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Modello generato da Colab (copia led_model_data.h in src/)
#include "led_model_data.h"

// ── Configurazione ───────────────────────────────────────────────────────────

// Imposta a true per output human-readable, false per JSON (es. parsing Python)
#define VERBOSE_OUTPUT      false

// Soglia minima di confidenza per considerare valida una predizione
#define CONFIDENCE_THRESHOLD 0.70f

// Dimensioni immagine attese dal modello (devono corrispondere al training)
#define IMG_WIDTH   96
#define IMG_HEIGHT  96
#define IMG_CHANNELS 3

// Memoria arena per TFLite Micro
// 96KB: dimensionato per la CNN con attivazioni INT8 + overhead
// Se ottieni un errore "AllocateTensors failed" aumenta di 8KB alla volta
#define TFLITE_ARENA_SIZE (512 * 1024) // 512 KB in PSRAM

// ── Pin camera Freenove WROVER ───────────────────────────────────────────────
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  21
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    19
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM     5
#define Y2_GPIO_NUM     4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// ── Label classi (ordine identico al training) ───────────────────────────────
const char* CLASS_LABELS[] = {"red", "green", "blue", "no_led"};
const int   NUM_CLASSES    = 4;

// ── Variabili globali TFLite ─────────────────────────────────────────────────
namespace {
    // Arena di memoria statica per TFLite Micro (in DRAM)
    uint8_t* tflite_arena = nullptr;

    const tflite::Model*        model       = nullptr;
    tflite::MicroInterpreter*   interpreter = nullptr;
    TfLiteTensor*               input       = nullptr;
    TfLiteTensor*               output      = nullptr;

    // Statistiche runtime
    uint32_t inference_count = 0;
    uint32_t total_ms        = 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Inizializzazione fotocamera
// ════════════════════════════════════════════════════════════════════════════

bool init_camera() {
    camera_config_t config;

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    // Clock: 20MHz stabile per RGB888 a 96x96
    config.xclk_freq_hz = 20000000;

    // RGB888: 3 byte/pixel — stesso formato del training
    config.pixel_format = PIXFORMAT_RGB565;

    // Risoluzione minima che supporta 96x96 nativamente
    config.frame_size   = FRAMESIZE_96X96;

    config.jpeg_quality = 10;   // non usato con RGB888, ma richiesto dalla struct
    config.fb_count     = 1;    // 1 frame buffer: riduce RAM, sufficiente per inferenza

    // PSRAM: usa fb in PSRAM se disponibile
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAMERA] Errore init: 0x%x\n", err);
        return false;
    }

    // Ottimizzazioni sensore OV2640 per classificazione colore LED
    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor) {
        sensor->set_whitebal(sensor, 0);       // AWB OFF: colori stabili e ripetibili
        sensor->set_awb_gain(sensor, 0);       // Auto gain bilanciamento OFF
        sensor->set_wb_mode(sensor, 0);        // Modalità WB: auto disabilitata
        sensor->set_exposure_ctrl(sensor, 1);  // AEC ON: adatta la luminosità
        sensor->set_aec2(sensor, 1);           // AEC DSP ON
        sensor->set_gain_ctrl(sensor, 1);      // AGC ON
        sensor->set_bpc(sensor, 1);            // Bad pixel correction ON
        sensor->set_wpc(sensor, 1);            // White pixel correction ON
        sensor->set_raw_gma(sensor, 1);        // Gamma correction ON
        sensor->set_lenc(sensor, 1);           // Lens correction ON
        sensor->set_saturation(sensor, 1);     // Lieve boost saturazione (↑ discriminazione colori)
        sensor->set_contrast(sensor, 1);       // Contrasto leggermente aumentato
    }

    Serial.println("[CAMERA] Inizializzata ✓  RGB888 @ 96x96");
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Inizializzazione TFLite Micro
// ════════════════════════════════════════════════════════════════════════════

bool init_tflite() {

    if (tflite_arena == nullptr) {
        tflite_arena = (uint8_t*)heap_caps_malloc(TFLITE_ARENA_SIZE, MALLOC_CAP_SPIRAM);
        if (tflite_arena == nullptr) {
            Serial.println("[FATAL] Impossibile allocare la Tensor Arena in PSRAM!");
            return false;
        }
    }

    // 1. Carica il modello dall'array C (led_model_data.h)
    model = tflite::GetModel(led_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLITE] Schema version mismatch: %d vs %d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // 2. AllOpsResolver: include tutti gli operatori standard
    //    Per un footprint minore si può usare MicroMutableOpResolver
    //    e registrare solo: DepthwiseConv2D, Conv2D, MaxPool2D,
    //    FullyConnected, Softmax, Reshape, Mean (GlobalAveragePooling)
    static tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // 3. Crea l'interpreter con l'arena statica
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tflite_arena, TFLITE_ARENA_SIZE, error_reporter
    );
    interpreter = &static_interpreter;

    // 4. Alloca i tensori nell'arena
    TfLiteStatus alloc_status = interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("[TFLITE] AllocateTensors FALLITO");
        Serial.println("         → Aumenta TFLITE_ARENA_SIZE di 8KB");
        return false;
    }

    // 5. Ottieni i puntatori ai tensori di input/output
    input  = interpreter->input(0);
    output = interpreter->output(0);

    // 6. Verifica le dimensioni (sanity check)
    bool shape_ok = (input->dims->size    == 4         &&
                     input->dims->data[0] == 1         &&   // batch
                     input->dims->data[1] == IMG_HEIGHT &&  // H
                     input->dims->data[2] == IMG_WIDTH  &&  // W
                     input->dims->data[3] == IMG_CHANNELS); // C

    if (!shape_ok) {
        Serial.printf("[TFLITE] Input shape inattesa: [%d,%d,%d,%d]\n",
                      input->dims->data[0], input->dims->data[1],
                      input->dims->data[2], input->dims->data[3]);
        return false;
    }

    // Report memoria
    size_t used_bytes = interpreter->arena_used_bytes();
    Serial.printf("[TFLITE] Inizializzato ✓\n");
    Serial.printf("         Arena usata  : %u / %u bytes (%.1f%%)\n",
                  used_bytes, TFLITE_ARENA_SIZE,
                  (float)used_bytes / TFLITE_ARENA_SIZE * 100.0f);
    Serial.printf("         Input dtype  : %s\n",
                  input->type == kTfLiteFloat32 ? "float32" : "int8");
    Serial.printf("         Output shape : [1, %d]\n", output->dims->data[1]);

    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Preprocessing: frame buffer → tensore di input
// ════════════════════════════════════════════════════════════════════════════

/**
 * Copia il frame RGB888 nel tensore di input TFLite,
 * normalizzando i pixel da [0,255] a [0.0, 1.0].
 *
 * NOTA: la fotocamera produce RGB, il modello è stato addestrato su immagini
 * che OpenCV ha convertito da RGB a BGR e poi salvato come PNG.
 * cv2.imwrite() salva BGR → PNG, cv2.imread() rilegge BGR.
 * Il training usa tf.image.decode_png() che legge come RGB.
 * Quindi il modello si aspetta RGB — nessuna conversione necessaria qui.
 */
void preprocess_frame(const uint8_t* fb_data) {
    float* input_data = input->data.f;  // Tensore di input Float32

    const int total_pixels = IMG_WIDTH * IMG_HEIGHT;

    for (int i = 0; i < total_pixels; i++) {
        // L'OV2640 in RGB565 invia 2 byte per pixel (Big Endian)
        uint16_t pixel = (fb_data[i * 2] << 8) | fb_data[i * 2 + 1];

        // Estraiamo i singoli canali (5 bit per Rosso, 6 per Verde, 5 per Blu)
        uint8_t r = (pixel >> 11) & 0x1F;
        uint8_t g = (pixel >> 5)  & 0x3F;
        uint8_t b =  pixel        & 0x1F;

        // Normalizziamo direttamente da 0.0 a 1.0 per l'intelligenza artificiale
        input_data[i * 3 + 0] = (float)r / 31.0f;  // Il max del rosso è 31
        input_data[i * 3 + 1] = (float)g / 63.0f;  // Il max del verde è 63
        input_data[i * 3 + 2] = (float)b / 31.0f;  // Il max del blu è 31
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Inferenza e parsing del risultato
// ════════════════════════════════════════════════════════════════════════════

struct Prediction {
    int   class_idx;
    float confidence;
    bool  above_threshold;
};

Prediction run_inference() {
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
        return {-1, 0.0f, false};
    }

    // Output: vettore di probabilità softmax [red, green, blue, no_led]
    float* probs = output->data.f;

    int   best_idx  = 0;
    float best_prob = probs[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_idx  = i;
        }
    }

    return {
        best_idx,
        best_prob,
        best_prob >= CONFIDENCE_THRESHOLD
    };
}

// ════════════════════════════════════════════════════════════════════════════
// Output seriale
// ════════════════════════════════════════════════════════════════════════════

void print_result(const Prediction& pred, uint32_t elapsed_ms) {
    const char* label = pred.above_threshold
                        ? CLASS_LABELS[pred.class_idx]
                        : "uncertain";

    if (VERBOSE_OUTPUT) {
        // Output leggibile per debug
        Serial.printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        Serial.printf("  Classe      : %s\n", label);
        Serial.printf("  Confidenza  : %.1f%%\n", pred.confidence * 100.0f);
        Serial.printf("  Latenza     : %u ms\n", elapsed_ms);
        Serial.printf("  Inferenze   : %u  (media: %u ms)\n",
                      inference_count,
                      inference_count > 0 ? total_ms / inference_count : 0);

        if (pred.above_threshold) {
            // Stampa distribuzione completa delle probabilità
            float* probs = output->data.f;
            for (int i = 0; i < NUM_CLASSES; i++) {
                Serial.printf("  [%s]%s %.1f%%\n",
                    CLASS_LABELS[i],
                    (i == pred.class_idx) ? " ◄" : "  ",
                    probs[i] * 100.0f);
            }
        }
    } else {
        // Output JSON compatto — facile da parsare lato Python/Node
        float* probs = output->data.f;
        Serial.printf("{\"class\":\"%s\","
                       "\"confidence\":%.3f,"
                       "\"probs\":[%.3f,%.3f,%.3f,%.3f],"
                       "\"ms\":%u}\n",
                      label,
                      pred.confidence,
                      probs[0], probs[1], probs[2], probs[3],
                      elapsed_ms);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n╔══════════════════════════════════════╗");
    Serial.println("║   LED Classifier — TFLite Micro      ║");
    Serial.println("║   ESP32-CAM Freenove WROVER           ║");
    Serial.println("╚══════════════════════════════════════╝");

    // Inizializza camera
    if (!init_camera()) {
        Serial.println("[FATAL] Camera init fallita. Riavvio...");
        delay(3000);
        ESP.restart();
    }

    // Warm-up camera: scarta i primi 10 frame
    // L'OV2640 ha bisogno di qualche frame per stabilizzare AEC/AGC
    Serial.print("[CAMERA] Warm-up");
    for (int i = 0; i < 10; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(50);
        Serial.print(".");
    }
    Serial.println(" ✓");

    // Inizializza TFLite
    if (!init_tflite()) {
        Serial.println("[FATAL] TFLite init fallita. Riavvio...");
        delay(3000);
        ESP.restart();
    }

    // Info memoria
    Serial.printf("\n[MEM] Free heap       : %u bytes\n", ESP.getFreeHeap());
    Serial.printf("[MEM] Free PSRAM      : %u bytes\n", ESP.getFreePsram());
    Serial.printf("[MEM] Arena TFLite    : %u bytes\n", TFLITE_ARENA_SIZE);
    Serial.printf("[CFG] Soglia conf.    : %.0f%%\n\n", CONFIDENCE_THRESHOLD * 100.0f);

    Serial.println("Sistema pronto. Avvio inferenza continua...\n");
}

// ════════════════════════════════════════════════════════════════════════════
// LOOP
// ════════════════════════════════════════════════════════════════════════════

void loop() {
    // ── 1. Cattura frame ─────────────────────────────────────────────────────
    uint32_t t_start = millis();

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[WARN] Frame buffer null, skip");
        delay(100);
        return;
    }

    // Verifica dimensioni frame (sicurezza)
    const size_t expected_size = IMG_WIDTH * IMG_HEIGHT * 2;
    if (fb->len != expected_size) {
        Serial.printf("[WARN] Frame size inattesa: %u (attesa %u)\n",
                      fb->len, expected_size);
        esp_camera_fb_return(fb);
        return;
    }

    // ── 2. Preprocessing ─────────────────────────────────────────────────────
    preprocess_frame(fb->buf);

    // Rilascia subito il frame buffer (libera RAM/PSRAM)
    esp_camera_fb_return(fb);

    // ── 3. Inferenza ─────────────────────────────────────────────────────────
    Prediction pred = run_inference();

    uint32_t elapsed = millis() - t_start;

    if (pred.class_idx < 0) {
        Serial.println("[ERROR] Invoke fallito");
        return;
    }

    // ── 4. Aggiorna statistiche ───────────────────────────────────────────────
    inference_count++;
    total_ms += elapsed;

    // ── 5. Output ─────────────────────────────────────────────────────────────
    print_result(pred, elapsed);

    // Nessun delay artificiale: la camera è il collo di bottiglia naturale
    // Il loop gira a ~8-12 FPS con RGB888 96x96 + inferenza INT8
}