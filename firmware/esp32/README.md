# ESP32 — environmental telemetry → ThingSpeak

Firmware lives here so hardware and Python **channel field layouts** stay aligned
(`configs/thingspeak_channels.yaml`).

## Requirements

- **Arduino IDE 2.x** or **PlatformIO** with ESP32 board support
- Libraries: **WiFi**, **HTTPClient** (built-in), optional official **ThingSpeak** library
- **Write interval:** ≥ **15 seconds** between ThingSpeak updates (free tier). Prefer **900 s** (15 min).

## Sketch

See `agroedge_thingspeak_env/agroedge_thingspeak_env.ino`:

1. Set `SECRET_SSID`, `SECRET_PASS`, `TS_CHANNEL_ID`, `TS_WRITE_API_KEY`.
2. Adjust sensor reads where marked (GPIO / I2C / analog).
3. Map variables to **field1–field7** exactly as in `thingspeak_channels.yaml`.

## Irrigation actuation

The Pi + Python stack issues decisions; the ESP32 can subscribe to a **separate**
command path (MQTT, second ThingSpeak channel, or GPIO) in a follow-on sketch.
Do **not** open valves from the public dashboard.
