#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

int angles[5] = {0,0,0,0,0};
const int GRIP_OPEN = 10;
const int GRIP_CLOSE = 73;

void setup() {
  //put your setup code here, to run once:
  Braccio.begin();
  Braccio.ServoMovement(20, 90, 90, 90, 90, 90, 73);
  Serial.begin(9600);
}

void loop() {
  //put your main code here, to run repeatedly:
  if (Serial.available() > 0)
  {
    String data = Serial.readStringUntil('\n');
    update_angles(data);
    Braccio.ServoMovement(100, angles[0], angles[1], angles[2], angles[3], angles[4], GRIP_CLOSE);
  }
}

//function used to update angles based on string read
void update_angles(String data)
{ 
  int length = data.length();
  int i = 0; //index for string

  //outer for loop iterates through angles
  for (int j = 0; j < 5; j++)
  {
    //inner for loop iterates through data string
    for (int k = i + 1; k < length; k++)
    {
      if (data[k] == ',')
      { 
        if (k - i == 1) //single digit
        {
          char q = data[i]; //extract angle from string
          angles[j] = q - '0'; //convert to int + update angles array
        }
        else
        {
          String q = data.substring(i,k); //extract angle from string
          angles[j] = q.toInt(); //convert to int + update angles array
        }
        i = k + 1; //update index
        break;
      }
    }
  }
}

