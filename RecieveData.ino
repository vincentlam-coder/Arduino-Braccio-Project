#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

int pickup_angles[5] = {90,90,90,90,90};
int putdown_angles[5] = {90,90,90,90,90};
const int GRIP_OPEN = 10;
const int GRIP_CLOSE = 73;

void setup() {
  //put your setup code here, to run once:
  Braccio.begin();
  Braccio.ServoMovement(20, 90, 90, 90, 90, 90, GRIP_CLOSE);
  Serial.begin(9600);
}

void loop() {
  //put your main code here, to run repeatedly:
  if (Serial.available() > 0)
  { 
    delay(1000);
    String flag = Serial.readStringUntil('\n');

    if (flag == "P") //pickup + putdown
    {
      //update pickup angles
      String pickup_data = Serial.readStringUntil('\n');
      update_angles(pickup_data,true);
      //update putdown angles
      String putdown_data = Serial.readStringUntil('\n');
      update_angles(putdown_data,false);

      Braccio.ServoMovement(100, pickup_angles[0], pickup_angles[1], pickup_angles[2], pickup_angles[3],pickup_angles[4], GRIP_OPEN); //move to pickup position
      delay(500);
      Braccio.ServoMovement(100, pickup_angles[0], pickup_angles[1], pickup_angles[2], pickup_angles[3], pickup_angles[4], GRIP_CLOSE); //close gripper
      delay(500);
      Braccio.ServoMovement(100, 90, 90, 90, 90, 90, GRIP_CLOSE); //return to home configuration
      delay(500);
      Braccio.ServoMovement(100, putdown_angles[0], putdown_angles[1], putdown_angles[2], putdown_angles[3], putdown_angles[4], GRIP_CLOSE); //move to putdown position
      delay(500);
      Braccio.ServoMovement(100, putdown_angles[0], putdown_angles[1], putdown_angles[2], putdown_angles[3], putdown_angles[4], GRIP_OPEN); //open gripper
      delay(500);
      Braccio.ServoMovement(100, 90, 90, 90, 90, 90, GRIP_OPEN); //return to home configuration
    }
    else //other operations
    {
      String data = Serial.readStringUntil('\n');
      update_angles(data,true);
      Braccio.ServoMovement(100, pickup_angles[0], pickup_angles[1], pickup_angles[2], pickup_angles[3], pickup_angles[4], GRIP_CLOSE);
    }
  }
}

//function used to update pickup/putdown angles based on string read given flag variable
void update_angles(String data, bool flag)
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

          //convert to int + update pickup/putdown angles array based on flag
          if (flag == true)
          {
            pickup_angles[j] = q - '0'; 
          }
          else
          {
            putdown_angles[j] = q - '0';
          }
        }
        else
        {
          String q = data.substring(i,k); //extract angle from string

          //convert to int + update pickup/putdown angles array based on flag 
          if (flag == true)
          {
            pickup_angles[j] = q.toInt(); 
          }
          else
          {
            putdown_angles[j] = q.toInt(); 
          }
        }
        i = k + 1; //update index
        break;
      }
    }
  }
}

