#include <Arduino.h>
#include <BLEMidi.h>
int input = 0;

void setup() {
  Serial.begin(115200);
  BLEMidiServer.begin("HyVibeControl");
}

void loop() {
    if (Serial.available()) {
    input = Serial.readString().toInt();
    if(input == 1){
      BLEMidiServer.noteOff(0, 2, 127);
    }
    }
}
