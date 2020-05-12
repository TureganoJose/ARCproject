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
Joystick joystick("/dev/input/js1");
// Ensure that it was found and that we can use it
if (!joystick.isFound())
{
  printf("open failed.\n");
  return -1;
}
JoystickEvent event;


// Initialise GPIO
// https://raspberrypi.stackexchange.com/questions/56003/pigpio-servo-control
if (gpioInitialise() < 0)
{
    printf("GPIO didn't initialise \n");
    return 1;
}



Mat frame;


 // Begin of parallel region
 #pragma omp parallel sections default(shared) private(frame)
{

   #pragma omp section
   {
       printf("Running controller \n");


       // Cotnrolling joystick
       while (true)
       {
         // Restrict rate
         usleep(1000);

         // Attempt to sample an event from the joystick

         if (joystick.sample(&event))
         {
           if (event.isButton())
           {
             printf("Button %u is %s\n",
               event.number,
               event.value == 0 ? "up" : "down");
           }
           else if (event.isAxis())
           {
             printf("Axis %u is at position %d\n", event.number, event.value);
           }
         }
       }
   }

   #pragma omp section
   {
       int icounter = 0;

       while(1) {
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



gpioTerminate();

//destroyAllWindows();
cout << "bye!" <<endl;
}



