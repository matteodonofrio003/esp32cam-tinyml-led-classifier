#include <Arduino.h>
#include <esp_camera.h>
#include "../include/config.h"

// Function declarations
int myFunction(int, int);
bool initCamera();

void setup() {
  // Initialize serial communication
  Serial.begin(921600);
  delay(1000);
  
  Serial.println("\n\nStarting ESP32CAM TinyML LED Classifier");
  
  // Initialize the camera
  if (!initCamera()) {
    Serial.println("Camera initialization failed!");
    while (1) {
      delay(1000);
    }
  }
}

void loop() {
  // Put your main code here, to run repeatedly:
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame acquisition ERROR!");
    return; 
  }

  // Sending header: 2 bytes for sync
  Serial.write(0xAA);
  Serial.write(0xBB);
  // Sending dimension: 4 bytes
  Serial.write((fb->len >> 24) & 0xFF);
  Serial.write((fb->len >> 16) & 0xFF);
  Serial.write((fb->len >> 8) & 0xFF);
  Serial.write(fb->len & 0xFF);

  // Sending frame data
  Serial.write(fb->buf, fb->len);
  // Sending footer: 2 bytes
  Serial.write(0xCC);
  Serial.write(0xDD);
  // Releasing memory
  esp_camera_fb_return(fb);
  // 100 ms delay 
  delay(100);
  }

// Put function definitions here:

/**
 * Initializes the ESP32CAM camera
 * Configures the pins, frame buffer, and camera settings
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool initCamera() {
  // Camera configuration
  camera_config_t config;
  
  // Pin configuration
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  
  // Camera data pins
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d0 = Y2_GPIO_NUM;
  
  // Control pins
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  
  // Frequency and frame buffer settings
  config.xclk_freq_hz = 20000000;  // XCLK frequency 20MHz
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  
  // Frame buffer configuration
  config.frame_size = FRAMESIZE_96X96;    // 96×96
  config.jpeg_quality = 12;              // 0=max quality, 63=min
  config.fb_count = 1;                   // Number of frame buffers
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  
  // Pixel format settings
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