#include "Arduino.h"
#include "AX12A.h"

#define DirectionPin    (10u)
#define BaudRate        (1000000ul)

#define ID1            (1u)
#define ID2            (2u)
#define ID3            (3u)
#define ID4            (4u)
#define ID5            (5u)
#define ID6            (6u)

int mot1Pos = 512;
int mot2Pos = 512;
int mot3Pos = 512;
int mot4Pos = 956;
int mot5Pos = 800;
int mot6Pos = 430;


 
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  ax12a.begin(BaudRate, DirectionPin, &Serial1);
}

void loop() {
  // put your main code here, to run repeatedly:
  receiveSerial();
  moveMotor();
}

void moveMotor() {
  ax12a.move(ID1, mot1Pos);
  ax12a.move(ID2, mot2Pos);
  ax12a.move(ID3, mot3Pos);
  ax12a.move(ID4, mot4Pos);
  ax12a.move(ID5, mot5Pos);
  ax12a.move(ID6, mot6Pos);

  
  Serial.print(mot1Pos);
  Serial.print(",");
  Serial.print(mot2Pos);
  Serial.print(",");
  Serial.print(mot3Pos);
  Serial.print(",");
  Serial.print(mot4Pos);
  Serial.print(",");
  Serial.print(mot5Pos);
  Serial.print(",");
  Serial.print(mot6Pos);
  Serial.println();


}

void receiveSerial() {
  if (Serial.available()) {
    int n1, n2, n3, n4, n5, n6;
    String packet = Serial.readStringUntil('*');
    sscanf(packet.c_str(), "%d,%d,%d,%d,%d,%d", &n1, &n2, &n3, &n4, &n5, &n6); //mot1Pos,mot1Pos,mot1Pos,mot1Pos

    mot1Pos = constrain(n1, 0, 1023);
    mot2Pos = constrain(n2, 512, 634);
    mot3Pos = constrain(n3, 104, 924);
    mot4Pos = constrain(n4, 512, 956);
    mot5Pos = constrain(n4, 0, 1023);
    mot6Pos = constrain(n4, 430, 890);
  }
}
