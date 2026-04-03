#include <Arduino.h>
#include <esp_camera.h>
#include "../include/config.h"

// Function declarations
int myFunction(int, int);
bool initCamera();

void setup() {
  // Inizializza la comunicazione seriale
  Serial.begin(921600);
  delay(1000);
  
  Serial.println("\n\nAvvio ESP32CAM TinyML LED Classifier");
  
  // Inizializza la camera
  if (!initCamera()) {
    Serial.println("Fallimento inizializzazione camera!");
    while (1) {
      delay(1000);
    }
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame acquisition ERROR!");
    return; 
  }

  // Sending header: 2 bytes for sync
  Serial.write(0xAA);
  Serial.write(0xBB);
  // Seniding dimension: 4 bytes
  Serial.write((fb->len >> 24) & 0xFF);
  Serial.write((fb->len >> 16) & 0xFF);
  Serial.write((fb->len >> 8) & 0xFF);
  Serial.write(fb->len & 0xFF);

  // Sending frame data
  Serial.write(fb->buf, fb->len);
  // Sending footer: 2 bytes
  Serial.write(0xCC);
  Serial.write(0xDD);
  // Realising memory
  esp_camera_fb_return(fb);
  // 100 ms delay 
  delay(100);
  }

// Put function definitions here:

/**
 * Inizializza la camera ESP32CAM
 * Configura i pin, i frame buffer e le impostazioni della camera
 * 
 * @return true se l'inizializzazione è riuscita, false altrimenti
 */
bool initCamera() {
  // Configurazione della camera
  camera_config_t config;
  
  // Configurazione dei pin
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  
  // Pin dati camera
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d0 = Y2_GPIO_NUM;
  
  // Pin di controllo
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  
  // Impostazioni frequenza e frame buffer
  config.xclk_freq_hz = 20000000;  // XCLK frequenza 20MHz
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  
  // Configurazione frame buffer
  config.frame_size = FRAMESIZE_96X96;    // 96×96
  config.jpeg_quality = 12;              // 0=qualità massima, 63=minima
  config.fb_count = 1;                   // Numero frame buffer
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  
  // Impostazioni pixel format
  config.pixel_format = PIXFORMAT_RGB565; 
  
  // Inizializzazione camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Errore inizializzazione camera: 0x%x\n", err);
    return false;
  }
  
  // Configurazione aggiuntiva dei parametri della camera
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    // Impostazioni di controllo automatico
    s->set_brightness(s, 0);       // Luminosità (da -2 a 2)
    s->set_contrast(s, 0);         // Contrasto (da -2 a 2)
    s->set_saturation(s, 0);       // Saturazione (da -2 a 2)
    
    // Controllo automatico
    s->set_ae_level(s, 0);         // Auto exposure level (da -2 a 2)
    s->set_aec2(s, false);         // Auto exposure compression
    s->set_awb_gain(s, true);      // Guadagno bilanciamento bianco
    s->set_agc_gain(s, 0);         // Guadagno controllo guadagno auto (0=auto)
    
    // Specchio/Flip
    s->set_hmirror(s, 0);          // Flip orizzontale
    s->set_vflip(s, 0);            // Flip verticale
  }
  
  Serial.println("Camera inizializzata correttamente!");
  return true;
}

int myFunction(int x, int y) {
  return x + y;
}