#include <iostream>
#include <vector>

// OpenMP header
// #include <omp.h>
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

// Tensorflow lite
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace cv;
using namespace std;



int main()
{

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("/home/pi/Documents/ARCproject/ARC02_inference/tflite_HairMatteNet.tflite"); //tflite_HairMatteNet simple_model_v2 HairMatteNet_q C_UNETplusplus
    if(model!=nullptr)
    {
        printf("Loaded model \n");
    }
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (interpreter != nullptr)
    {
        printf("Loaded interpreter \n");
    }

    // Initialise camera
    VideoCapture cap(0);
//    if (!cap.isOpened()) {
//      printf("ERROR: Unable to open the camera \n");
//      cap.release();
//      return -1;
//    }
//    else
//    {
        cap.set(CAP_PROP_FRAME_WIDTH,1024); // 1024  640
        cap.set(CAP_PROP_FRAME_HEIGHT,768); // 768   480
        printf("Camera is open \n");
//    }
    Mat unrotated_frame;
    Mat frame;

    const int output_size[] = {1, 224, 224};
    const int mask_size[] = {224, 224};

    // Create an instance of Joystick
    Joystick joystick("/dev/input/js1");
    // Ensure that it was found and that we can use it
    if (!joystick.isFound())
    {
      printf("open failed.\n");
      //return -1;
    }
    JoystickEvent event;
    bool exit_flag = false;

    // Initialise GPIO
    // https://raspberrypi.stackexchange.com/questions/56003/pigpio-servo-control
    if (gpioInitialise() < 0)
    {
        printf("GPIO didn't initialise \n");
        //return 1;
    }

    int ESC_gpio = 4; // GPIO 4 (pin 7)
    int servo_gpio = 17; // GPIO 17 (pin 11)
    double max_servo_value = 1780; //Servo's max value
    double min_servo_value = 980;  //Servo's min value
    double max_esc_value = 2000; //ESC's max value
    double min_esc_value = 1500;  //ESC's min value
    double zero_value;
    double half_range;

    zero_value = (max_servo_value + min_servo_value)/2;
    half_range = (max_servo_value - min_servo_value)/2;

    double steering = 0.0;
    double speed = 0.0;

    double speed_incr= 5.0;
    double steering_incr = 5.0;

    gpioServo(servo_gpio, steering);
    gpioServo(ESC_gpio, speed);



    // Main loop camera
    while(!exit_flag)
    {
        // Image source from camera
        cap >> unrotated_frame;

        // Image source from file to test
        //frame = cv::imread("/home/pi/Documents/ARCproject/ARC02_inference/20200520_103757Tick4542str1780.jpg");
        if (unrotated_frame.empty())
        {
            printf("ERROR: Unable to grab from the camera \n");
            break;
        }
        cv::flip(unrotated_frame, frame, -1);


        // Benchmark
        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // Resize image and normalise
        cv::resize(frame, frame, cv::Size(224,224));
        frame.convertTo(frame, CV_32FC3, 1.0 / 255, 0);
        //printf("Total len %d Element size %d \n", frame.total(), frame.elemSize1());
        //cout << "M = " << endl << " " << frame << endl << endl;

        // Allocate tensor buffers.
        interpreter->AllocateTensors();

        //int image_width = 224;
        //int image_height = 224;
        //int image_channels = 3;
        //int input = interpreter->inputs()[0];
        //TfLiteIntArray* dims = interpreter->tensor(input)->dims;
        //int wanted_images = dims->data[0];
        //int wanted_height = dims->data[1];
        //int wanted_width = dims->data[2];
        //int wanted_channels = dims->data[3];
        //printf("I %d H %d W %d C %d \n",wanted_images, wanted_height, wanted_width, wanted_channels);
        //int type = interpreter->tensor(input)->type;
        //printf("Type %d \n", type);
        // Types supported by tensor
//        typedef enum {
//          kTfLiteNoType = 0,
//          kTfLiteFloat32 = 1,
//          kTfLiteInt32 = 2,
//          kTfLiteUInt8 = 3,
//          kTfLiteInt64 = 4,
//          kTfLiteString = 5,
//          kTfLiteBool = 6,
//          kTfLiteInt16 = 7,
//          kTfLiteComplex64 = 8,
//          kTfLiteInt8 = 9,
//        } TfLiteType;


        // Fill input buffers
        // https://github.com/finnickniu/tensorflow_object_detection_tflite/blob/master/demo.cpp
        memcpy(interpreter->typed_input_tensor<float>(0), frame.data, frame.total() * frame.elemSize());

        // Run inference
        interpreter->Invoke();
        // Read output buffers
        // TODO(user): Insert getting data out code.
        //int output = interpreter->outputs()[0];
        //std::cout << interpreter->typed_output_tensor<float>(0) << std::endl;
        //float* output = interpreter->typed_output_tensor<float>(0);
        //int output = interpreter->outputs()[0];
        //TfLiteIntArray* dims_out = interpreter->tensor(output)->dims;
        //int output_height = dims_out->data[1];
        //int output_width = dims_out->data[2];
        //int output_channels = dims_out->data[3];
        //printf("Output H %d W %d C %d \n", output_height, output_width, output_channels);

        Mat output_frame(3,output_size, CV_32FC2);
        memcpy(output_frame.data, interpreter->typed_output_tensor<float>(0), output_frame.total() * output_frame.elemSize());

        // Testing one pixel, comparing to python code for the same input image
        //float* mp = &output_frame.at<float>(0,93,1);
        //for(int i=0;i<2;i++)
        //{
        //    printf("value %d is %lf",i,mp[i]);
        //}

        // Creating mask
        Mat mask_image(2,mask_size,CV_32F);
        mask_image = 0;
        for(int iheight=0;iheight<224;iheight++)
        {
            for(int iwidth=0;iwidth<224;iwidth++)
            {
                float* mp = &output_frame.at<float>(0,iheight,iwidth);
                if(mp[0]>0.5)
                {
                    mask_image.at<float>(iheight,iwidth) = 0.0;
                }
                else if(mp[1]>0.5)
                {
                    mask_image.at<float>(iheight,iwidth) = 1.0;
                }

            }
        }

        // Find index of maximum value in vector of sum columns
        Mat mask_image_summed;
        cv::reduce(mask_image, mask_image_summed, 0, 0, CV_32FC1);
        mask_image_summed.convertTo(mask_image_summed, CV_32SC1);
        int max_loc[3];
        double max_value;
        cv::minMaxIdx(mask_image_summed,0,&max_value,0,max_loc);
        // Checking that the sum of the columns is correct
        //cout << "M = " << endl << " " << mask_image_summed << endl << endl;
        //printf("max column %d \n",max_loc[1]);
        //printf("max value %f \n",max_value);

        double K_d = 1.0 * (max_loc[1]-112)/101;
        K_d = min(K_d,  1.0);
        K_d = max(K_d, -1.0);
        steering = zero_value + half_range * K_d;
        gpioServo(servo_gpio, steering);



        //namedWindow( "Display window", WINDOW_AUTOSIZE );
        imshow( "Display window", frame );
        waitKey(0);                                          // Wait for a keystroke in the window
        //return 0;



        // Joystick controls
        if (joystick.sample(&event))
         {


           if (event.isButton())
           {
            if(event.number==0 && event.value==1) // When pressing X in PS3 controller
            {
                gpioServo(ESC_gpio, 0.0);
                gpioServo(servo_gpio, 0.0);
                cap.release();
                printf("Bye! \n");
                gpioTerminate();
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
            else if(event.number==15 && event.value==1) //Arrow left in controller
            {

                zero_value -= steering_incr;
                steering = zero_value;
                gpioServo(servo_gpio, steering);
            }
            else if(event.number==16 && event.value==1) //Arrow right in controller
            {
                zero_value += steering_incr;
                steering = zero_value;
                gpioServo(servo_gpio, steering);
            }
              //printf("Button %u is %s\n",
              //event.number,
              //event.value == 0 ? "up" : "down");
           }
           else if (event.isAxis())
           {
            if(event.number==3)
            {
                steering = zero_value - half_range * (event.value/32767.0);
                //printf("controller %d steering %f division %f \n",event.value, steering,(event.value/32767.0));
                gpioServo(servo_gpio, steering);
            }
             //printf("Axis %u is at position %d\n", event.number, event.value);
           }
         }
        // Finish benchmark
        //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Inference time = " << std::chrono::duration_cast <std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

    }



    return 0;
}
