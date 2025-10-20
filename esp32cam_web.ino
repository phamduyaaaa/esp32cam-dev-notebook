#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// =================== WiFi Config ===================
const char* ssid = "I-robot Lab_5G";
const char* password = "irobotlab";
WebServer server(80);

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

bool collecting = false;

// =================== HTML Interface ===================
void handleRoot() {
  String html = R"rawliteral(
  <html>
  <head>
    <title>ESP32-CAM Image Collector</title>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <style>
      body { font-family: Arial; text-align: center; background-color: #f8f8f8; }
      h1 { color: #333; }
      button { padding: 15px 40px; font-size: 20px; border: none; border-radius: 10px; cursor: pointer; margin: 15px; }
      .on { background-color: green; color: white; }
      .off { background-color: red; color: white; }
      img { width: 90%; margin-top: 20px; border-radius: 10px; }
    </style>
  </head>
  <body>
    <h1>ESP32-CAM Image Collector</h1>
    <button id='btn' class='on' onclick='toggleCollect()'>START COLLECT</button>
    <img id='stream' src='/stream'>
    <script>
      let collecting = false;
      async function toggleCollect() {
        collecting = !collecting;
        document.getElementById('btn').textContent = collecting ? "STOP COLLECT" : "START COLLECT";
        document.getElementById('btn').className = collecting ? "off" : "on";
        fetch(collecting ? "/start_collect" : "/stop_collect");
      }
    </script>
  </body>
  </html>
  )rawliteral";

  server.send(200, "text/html", html);
}

// =================== Stream ===================
void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  unsigned long last_time = millis();
  int frame_count = 0;

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) continue;

    String header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + String(fb->len) + "\r\n\r\n";
    client.write(header.c_str(), header.length());
    client.write(fb->buf, fb->len);
    client.write("\r\n");

    esp_camera_fb_return(fb);
    frame_count++;

    if (millis() - last_time >= 1000) {
      Serial.printf("FPS: %d | Collecting: %d\n", frame_count, collecting);
      frame_count = 0;
      last_time = millis();
    }

    if (!client.connected()) break;
  }
}

// =================== Start/Stop Collect ===================
void handleStartCollect() {
  collecting = true;
  Serial.println("START collecting signal sent!");
  server.send(200, "text/plain", "Collecting started");
}

void handleStopCollect() {
  collecting = false;
  Serial.println("STOP collecting signal sent!");
  server.send(200, "text/plain", "Collecting stopped");
}

// =================== Setup ===================
void setup() {
  Serial.begin(115200);

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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    return;
  }

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/stream", handleStream);
  server.on("/start_collect", handleStartCollect);
  server.on("/stop_collect", handleStopCollect);

  server.begin();
  Serial.println("Server started!");
}

void loop() {
  server.handleClient();
}
