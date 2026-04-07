#include <Arduino.h>
#include "esp_camera.h"
#include "esp_heap_caps.h" // NOSTRO FIX: Per la PSRAM

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h" // NOSTRO FIX: Per l'errore di compilazione
#include "tensorflow/lite/schema/schema_generated.h"
#include "led_model_data.h"

#define CONFIDENCE_THRESHOLD  0.70f
#define IMG_WIDTH             96
#define IMG_HEIGHT            96
#define IMG_CHANNELS          3
#define TFLITE_ARENA_SIZE     (512 * 1024) // NOSTRO FIX: Mezzo MegaByte

#define STREAM_EVERY_N_FRAMES 1

// Pin Freenove WROVER
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

const char* CLASS_LABELS[] = {"red", "green", "blue", "no_led"};
const int   NUM_CLASSES    = 4;

const uint8_t FRAME_HEADER[2] = {0xAA, 0xBB};
const uint8_t FRAME_FOOTER[2] = {0xCC, 0xDD};

namespace {
    uint8_t* tflite_arena = nullptr; // NOSTRO FIX: Puntatore per PSRAM
    const tflite::Model*       model       = nullptr;
    tflite::MicroInterpreter*  interpreter = nullptr;
    TfLiteTensor*              input       = nullptr;
    TfLiteTensor*              output      = nullptr;
    uint32_t                   frame_count = 0;
}

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
    config.xclk_freq_hz = 20000000;
    
    // NOSTRO FIX: RGB565 per evitare il Kernel Panic
    config.pixel_format = PIXFORMAT_RGB565; 
    config.frame_size   = FRAMESIZE_96X96;
    config.jpeg_quality = 10;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) return false;

    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_whitebal(s, 1);       
        s->set_awb_gain(s, 1);
        s->set_wb_mode(s, 0);
        s->set_exposure_ctrl(s, 1);  
        s->set_aec2(s, 1);
        s->set_gain_ctrl(s, 1);      
        s->set_bpc(s, 1);
        s->set_wpc(s, 1);
        s->set_raw_gma(s, 1);
        s->set_lenc(s, 1);
        s->set_saturation(s, 1);     
        s->set_contrast(s, 1);
    }
    return true;
}

bool init_tflite() {
    // NOSTRO FIX: Allocazione in PSRAM
    if (tflite_arena == nullptr) {
        tflite_arena = (uint8_t*)heap_caps_malloc(TFLITE_ARENA_SIZE, MALLOC_CAP_SPIRAM);
        if (!tflite_arena) return false;
    }

    model = tflite::GetModel(led_model_data);
    
    // NOSTRO FIX: L'Error reporter per far felice il compilatore
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tflite_arena, TFLITE_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) return false;

    input  = interpreter->input(0);
    output = interpreter->output(0);
    return true;
}

void stream_frame(const uint8_t* data, size_t len) {
    Serial.write(FRAME_HEADER, 2);
    uint8_t sz[4] = {
        (uint8_t)((len >> 24) & 0xFF),
        (uint8_t)((len >> 16) & 0xFF),
        (uint8_t)((len >>  8) & 0xFF),
        (uint8_t)((len      ) & 0xFF)
    };
    Serial.write(sz, 4);

    const size_t CHUNK = 1024;
    size_t sent = 0;
    while (sent < len) {
        size_t to_send = min(CHUNK, len - sent);
        Serial.write(data + sent, to_send);
        sent += to_send;
    }
    Serial.write(FRAME_FOOTER, 2);
}

// NOSTRO FIX: La matematica per decodificare l'RGB565
void preprocess(const uint8_t* fb_data) {
    float* in = input->data.f;
    const int n = IMG_WIDTH * IMG_HEIGHT;
    for (int i = 0; i < n; i++) {
        uint16_t pixel = (fb_data[i * 2] << 8) | fb_data[i * 2 + 1];
        uint8_t r = (pixel >> 11) & 0x1F;
        uint8_t g = (pixel >> 5)  & 0x3F;
        uint8_t b =  pixel        & 0x1F;
        in[i*3+0] = (float)r / 31.0f;
        in[i*3+1] = (float)g / 63.0f;
        in[i*3+2] = (float)b / 31.0f;
    }
}

struct Prediction { int idx; float conf; };

Prediction run_inference() {
    interpreter->Invoke();
    float* p  = output->data.f;
    int best  = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
        if (p[i] > p[best]) best = i;
    return {best, p[best]};
}

void setup() {
    Serial.begin(460800);
    delay(300);

    if (!init_camera()) { delay(3000); ESP.restart(); }
    for (int i = 0; i < 8; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(40);
    }
    if (!init_tflite()) { delay(3000); ESP.restart(); }
}

void loop() {
    uint32_t t0 = millis();

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(5); return; }

    // NOSTRO FIX: Verifica la dimensione per RGB565 (2 byte)
    if (fb->len != (size_t)(IMG_WIDTH * IMG_HEIGHT * 2)) {
        esp_camera_fb_return(fb);
        return;
    }

    frame_count++;

    preprocess(fb->buf);
    Prediction pred = run_inference();
    uint32_t elapsed = millis() - t0;

    if (frame_count % STREAM_EVERY_N_FRAMES == 0) {
        stream_frame(fb->buf, fb->len);
    }
    esp_camera_fb_return(fb);

    const char* label = (pred.conf >= CONFIDENCE_THRESHOLD)
                        ? CLASS_LABELS[pred.idx] : "uncertain";

    float* p = output->data.f;
    Serial.printf("{\"class\":\"%s\",\"confidence\":%.3f,"
                  "\"probs\":[%.3f,%.3f,%.3f,%.3f],\"ms\":%u}\n",
                  label, pred.conf,
                  p[0], p[1], p[2], p[3], elapsed);
}