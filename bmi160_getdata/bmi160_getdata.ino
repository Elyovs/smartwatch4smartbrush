#include <DFRobot_BMI160.h>
#include <MadgwickAHRS.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// --- CALIBRATION VALUES (Plug your values here) ---
const float GYRO_OFF_X = 2.5; 
const float GYRO_OFF_Y = 18.28;
const float GYRO_OFF_Z = 0.35;
const float ACCEL_OFF_X = 478.20;
const float ACCEL_OFF_Y = -1602.75;
const float ACCEL_OFF_Z = -508.03; 

// --- Constants ---
#define GYRO_RESOLUTION 16.4
#define ACCEL_1G 16384.0
const int8_t i2c_addr = 0x69;
const float alpha = 0.7; // Filter coefficient

// --- WiFi Configuration ---
const char* ssid = "LAPTOP-5FMMIAKQ 5454";
const char* password = "782%7yQ8";
const char* ip = "172.31.219.244";
const int port = 4210;

// --- Objects ---
DFRobot_BMI160 bmi160;
Madgwick madgwick;
WiFiUDP udp;

float filteredGyro[3] = {0};
float filteredAccel[3] = {0};

void setup() {
  Serial.begin(115200);
  delay(1000);

  // 1. Connect WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");

  // 2. Init BMI160
  if (bmi160.softReset() != BMI160_OK) {
    Serial.println("Reset failed");
    while(1);
  }
  if (bmi160.I2cInit(i2c_addr) != BMI160_OK) {
    Serial.println("I2C Init failed");
    while(1);
  }
  Serial.println("BMI160 Ready");

  // 3. Init Filter
  madgwick.begin(100); // 100Hz
}

void loop() {
  static unsigned long lastTime = 0;
  unsigned long now = millis();
  
  // Maintain ~100Hz loop (10ms)
  if (now - lastTime < 10) return;
  lastTime = now;

  int16_t rawData[6] = {0}; 
  
  // Read Sensor (rawData[0-2] = Gyro, rawData[3-5] = Accel)
  if (bmi160.getAccelGyroData(rawData) == 0) {
    
    // 1. Apply Calibration & Convert to G-force
    // Formula: (Raw - Offset) / LSB per G
    float ax = (rawData[3] - ACCEL_OFF_X) / ACCEL_1G;
    float ay = (rawData[4] - ACCEL_OFF_Y) / ACCEL_1G;
    float az = (rawData[5] - ACCEL_OFF_Z) / ACCEL_1G;

    // 2. Apply Calibration & Convert to Degrees per Second (DPS)
    // Formula: (Raw - Offset) / LSB per DPS
    float gx = (rawData[0] - GYRO_OFF_X) / GYRO_RESOLUTION;
    float gy = (rawData[1] - GYRO_OFF_Y) / GYRO_RESOLUTION;
    float gz = (rawData[2] - GYRO_OFF_Z) / GYRO_RESOLUTION;

    // 3. Prepare UDP Packet (CSV Format: ax, ay, az, gx, gy, gz)
    char packet[128]; // Increased buffer size for safety
    snprintf(packet, sizeof(packet), "%.4f, %.4f, %.4f, %.4f, %.4f, %.4f", 
             ax, ay, az, gx, gy, gz);
    
    // 4. Send Packet
    udp.beginPacket(ip, port);
    udp.print(packet);
    udp.endPacket();

    // Debug to Serial Monitor
    Serial.print("Sent UDP: ");
    Serial.println(packet);
    
    // Note: If you still need the Madgwick filter for local LCD display, 
    // keep this line; otherwise, you can delete it to save processing power.
    madgwick.updateIMU(gx * (PI/180.0), gy * (PI/180.0), gz * (PI/180.0), ax, ay, az);
  }
}