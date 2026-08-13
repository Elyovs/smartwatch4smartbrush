#include <TFT_eSPI.h>       // Graphics Lib
#include "SensorQMI8658.hpp"  // Sensor Lib by Lewisxhe (Install via Library Manager)
#include <Wire.h>
#include <WiFi.h>

SensorQMI8658 qmi;
IMUdata acc;
IMUdata gyr;

// WiFi Configuration
const char* ssid = "LAPTOP-5FMMIAKQ 5454";
const char* password = "782%7yQ8";
const char* ip = "192.168.0.68";
// const char* ip = "172.31.219.178";
const int port = 4210;

WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  delay(1000); // Give you time to open the monitor
  while (!Serial && millis() < 3000) {
    delay(10);
  }
  Serial.println("1. System Starting...");

  // Connecting WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  udp.begin(port);

  // Wire.begin(6, 7); 
  Wire.begin(15, 14); 
  Serial.println("3. Wire Started");

  // Initialize IMU
  // Note: We pass 'Wire' but don't need to pass pins 6,7 again since we did it above
  if (!qmi.begin(Wire, QMI8658_L_SLAVE_ADDRESS)) {
    Serial.println("4. FAILED to find QMI8658");
    while (1) {
      delay(1000);
    }
  }
  Serial.println("4. IMU Found");

  // Configure Accel
  qmi.configAccelerometer(SensorQMI8658::ACC_RANGE_4G, SensorQMI8658::ACC_ODR_1000Hz, SensorQMI8658::LPF_MODE_0);
  qmi.enableAccelerometer();
  Serial.println("5. Accel Setup Complete");

  qmi.configGyroscope(SensorQMI8658::GYR_RANGE_64DPS, SensorQMI8658::GYR_ODR_896_8Hz, SensorQMI8658::LPF_MODE_3);
  qmi.enableGyroscope();
  Serial.println("6. Gyro Setup Complete");
}

void loop() {
  if (qmi.getDataReady()) {
    char packet[64];

    if (qmi.getAccelerometer(acc.x, acc.y, acc.z)) {
      Serial.println("Accelerometer");
      Serial.println(acc.x);
      Serial.println(acc.y);
      Serial.println(acc.z);
      Serial.println();
    }
    if (qmi.getGyroscope(gyr.x, gyr.y, gyr.z)) {
      Serial.println("Gyroscope");
      Serial.println(gyr.x);
      Serial.println(gyr.y);
      Serial.println(gyr.z);
    }

    snprintf(packet, sizeof(packet), "%.2f, %.2f, %.2f, %.2f, %.2f, %.2f", 
              acc.x, acc.y, acc.z, gyr.x, gyr.y, gyr.z);
    
    // Send UDP Packet
    udp.beginPacket(ip, port);
    udp.print(packet);
    udp.endPacket();
  }
  delay(10);
}