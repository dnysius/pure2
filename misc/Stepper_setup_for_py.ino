#include <Adafruit_MotorShield.h>
#include <Wire.h>

// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 
// Or, create it with a different I2C address (say for stacking)
// Adafruit_MotorShield AFMS = Adafruit_MotorShield(0x61); 

// Connect a stepper motor with 200 steps per revolution (1.8 degree)
// to motor port #2 (M3 and M4)
Adafruit_StepperMotor *myMotor1 = AFMS.getStepper(200, 1);
Adafruit_StepperMotor *myMotor2 = AFMS.getStepper(200, 2);

  int totalcount1 = 0;           //initializing position tracker to 0 
  int totalcount2 = 0;           //initializing position tracker to 0 
  
void setup() {
 
  myMotor1 -> setSpeed(10);  // 10 rpm   
  myMotor2 -> setSpeed(10);  // 10 rpm   
  while (! Serial);             //makes sure the serial printing occurs when serial is running 
  Serial.begin(9600);           // set up Serial library at 9600 bps
  AFMS.begin();  // create with the default frequency 1.6KHz
  //AFMS.begin(1000);  // OR with a different frequency, say 1KHz  
}


void loop() {
      if (Serial.available()) {
        char serialListener = Serial.read();
        if (serialListener == '1') {
          myMotor1 -> step(40, FORWARD, DOUBLE); //moves the user specified number of steps    
          }
        else if (serialListener == '2') {
          myMotor1 -> step(40, BACKWARD, DOUBLE); //moves the user specified number of steps    
          }
       else if (serialListener == '3') {
          myMotor2 -> step(40, FORWARD, DOUBLE); //moves the user specified number of steps    
          }
       else if (serialListener == '4') {
          myMotor2 -> step(40, BACKWARD, DOUBLE); //moves the user specified number of steps    
       }
      }
}
