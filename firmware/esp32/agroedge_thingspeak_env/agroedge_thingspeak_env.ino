/**
 * AgroEdge — ESP32 environmental publish to ThingSpeak (template)
 *
 * Field mapping MUST match configs/thingspeak_channels.yaml (rice_env):
 *   field1 soil_moisture_percent
 *   field2 soil_temperature_celsius
 *   field3 air_temperature_celsius
 *   field4 air_humidity_percent
 *   field5 light_intensity_lux
 *   field6 water_tank_level_percent
 *   field7 water_flow_rate_lph (optional, 0 if unused)
 *
 * Install ESP32 board support; select your board and port.
 * ThingSpeak minimum interval: 15 s (use >= 900000 ms for rice monitoring).
 */

#include <WiFi.h>
#include <HTTPClient.h>

// --- WiFi (use build flags or replace before flashing) ---
const char *SECRET_SSID = "YOUR_WIFI_SSID";
const char *SECRET_PASS = "YOUR_WIFI_PASSWORD";

// --- ThingSpeak ---
const char *TS_WRITE_API_KEY = "YOUR_CHANNEL_WRITE_API_KEY";
const long TS_CHANNEL_ID = 0;  // numeric channel id

// Poll interval (ms). 900000 = 15 minutes.
const unsigned long UPDATE_INTERVAL_MS = 900000UL;

const char *TS_HOST = "api.thingspeak.com";

unsigned long lastSend = 0;

float read_soil_moisture_percent() {
  // TODO: map ADC / I2C sensor to 0–100 %
  return 45.0f;
}

float read_soil_temperature_celsius() {
  return 28.0f;
}

float read_air_temperature_celsius() {
  return 30.0f;
}

float read_air_humidity_percent() {
  return 65.0f;
}

float read_light_lux() {
  return 35000.0f;
}

float read_tank_level_percent() {
  return 75.0f;
}

float read_flow_lph() {
  return 0.0f;
}

bool sendThingspeak() {
  HTTPClient http;
  String url = String("https://") + TS_HOST + "/update";
  url += "?api_key=" + String(TS_WRITE_API_KEY);
  url += "&field1=" + String(read_soil_moisture_percent(), 2);
  url += "&field2=" + String(read_soil_temperature_celsius(), 2);
  url += "&field3=" + String(read_air_temperature_celsius(), 2);
  url += "&field4=" + String(read_air_humidity_percent(), 2);
  url += "&field5=" + String(read_light_lux(), 1);
  url += "&field6=" + String(read_tank_level_percent(), 2);
  url += "&field7=" + String(read_flow_lph(), 2);

  http.begin(url);
  int code = http.GET();
  http.end();
  return code > 0 && code != 401 && code != 403;
}

void setup() {
  Serial.begin(115200);
  delay(200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(SECRET_SSID, SECRET_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi OK");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSend < UPDATE_INTERVAL_MS) {
    delay(200);
    return;
  }
  lastSend = now;

  if (TS_CHANNEL_ID <= 0 || String(TS_WRITE_API_KEY).indexOf("YOUR_") >= 0) {
    Serial.println("Configure TS_CHANNEL_ID and TS_WRITE_API_KEY");
    return;
  }

  if (sendThingspeak()) {
    Serial.println("ThingSpeak update sent");
  } else {
    Serial.println("ThingSpeak update failed");
  }
}
