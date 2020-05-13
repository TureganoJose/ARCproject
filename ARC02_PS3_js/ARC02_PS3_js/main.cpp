// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int OMP_NUM_THREADS=4;

// Joystick header
#include "joystick.h"
#include <unistd.h>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ctime>
#include <vector>

// Raspberry Pi
#include <pigpio.h>
#include <algorithm>    // std::max std::min


using namespace cv;
using namespace std;



int main(int argc, char* argv[])
{

// Initialise camera
VideoCapture cap(0);
if (!cap.isOpened()) {
  printf("ERROR: Unable to open the camera \n");
  cap.release();
  return -1;
}
else
{
    cap.set(CAP_PROP_FRAME_WIDTH,1024);
    cap.set(CAP_PROP_FRAME_HEIGHT,768);
    printf("Camera is open \n");
}


// Create an instance of Joystick
Joystick joystick("/dev/input/js0");
// Ensure that it was found and that we can use it
if (!joystick.isFound())
{
  printf("open failed.\n");
  return -1;
}
JoystickEvent event;
bool exit_flag = false;

// Initialise GPIO
// https://raspberrypi.stackexchange.com/questions/56003/pigpio-servo-control
if (gpioInitialise() < 0)
{
    printf("GPIO didn't initialise \n");
    return 1;
}

int ESC_gpio = 4; // GPIO 4 (pin 7)
int servo_gpio = 17; // GPIO 17 (pin 11)
double max_servo_value = 1780; //Servo's max value
double min_servo_value = 980;  //Servo's min value
double max_esc_value = 2000; //ESC's max value
double min_esc_value = 980;  //ESC's min value
double zero_value;
double half_range;

zero_value = (max_servo_value + min_servo_value)/2;
half_range = (max_servo_value - min_servo_value)/2;



double steering = 0.0;
double speed = 0.0;

double speed_incr= 50.0;

gpioServo(servo_gpio, steering);
gpioServo(ESC_gpio, speed);

Mat frame;
 // Begin of parallel region
 #pragma omp parallel sections default(shared) private(frame)
{

   #pragma omp section
   {
       printf("Running controller \n");


       // Cotnrolling joystick
       while (!exit_flag)
       {
         // Restrict rate
         usleep(1000);

         // Attempt to sample an event from the joystick

         if (joystick.sample(&event))
         {


           if (event.isButton())
           {
            if(event.number==0 && event.value==1) // When pressing X in PS3 controller
            {
                printf("Bye! \n");
                exit_flag = true;
            }
            else if(event.number==9 && event.value==1) // Press start
            {
                speed = 1500;
            }
            else if(event.number==13 && event.value==1) //Arrow up in controller
            {
                speed += speed_incr;
                speed = max(speed,min_esc_value);
                speed = min(speed,max_esc_value);
                gpioServo(ESC_gpio, speed);
            }
            else if(event.number==14 && event.value==1) //Arrow down in controller
            {
                speed -= speed_incr;
                speed = max(speed,min_esc_value);
                speed = min(speed,max_esc_value);
                gpioServo(ESC_gpio, speed);
            }
              printf("Button %u is %s\n",
              event.number,
              event.value == 0 ? "up" : "down");
           }
           else if (event.isAxis())
           {
            if(event.number==3)
            {
                steering = zero_value + half_range * (event.value/32767.0);
                //printf("controller %d steering %f division %f \n",event.value, steering,(event.value/32767.0));
                gpioServo(servo_gpio, steering);
            }
             printf("Axis %u is at position %d\n", event.number, event.value);
           }
         }
       }
   }

   #pragma omp section
   {
       int icounter = 0;

       while(!exit_flag) {
         cap >> frame;
         if (frame.empty()) {
             printf("ERROR: Unable to grab from the camera \n");
             break;
         }
         bool result = false;

         // Saving frame
         try
         {
             time_t rawtime;
             struct tm * timeinfo;
             char buffer[80];

             time (&rawtime);
             timeinfo = localtime(&rawtime);
             strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
             std::string ctr_str = std::to_string(icounter);
             std::string date_str(buffer);
             std::string steering_str = std::to_string(event.value);
             result = imwrite("Tick"+ctr_str+"Data"+date_str+"str"+steering_str+".jpg", frame);
         }
         catch (const cv::Exception& ex)
         {
             fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
         }
         if (result)
         {
             icounter += 1;
             //printf("Saved JPG file with alpha data.\n");
         }
         else
         {
             printf("ERROR: Can't save JPG file.\n");
         }


           if(icounter==500)
           {
             break;
             cout << "Closing the camera" << endl;
             cap.release();

            }
       }
   }

}

if(event.number==0 && event.value==1)
{
    gpioServo(ESC_gpio, 0.0);
    gpioServo(servo_gpio, 0.0);
    printf("Bye! \n");
    //return 0;
}

gpioTerminate();
return 0;
//destroyAllWindows();
}



