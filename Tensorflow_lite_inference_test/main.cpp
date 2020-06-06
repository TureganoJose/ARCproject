#include <iostream>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ctime>
#include <vector>

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

    Mat unrotated_frame;
    Mat frame;

    const int output_size[] = {1, 224, 224};
    const int mask_size[] = {224, 224};


    // Image source from file to test
    //frame = cv::imread("/home/pi/Documents/ARCproject/ARC02_inference/20200520_103757Tick4542str1780.jpg");
    unrotated_frame = cv::imread("/media/pi/UUI/Late_evening_4th_test/20200601_202720Tick622str1380.000000.jpg");
    if (unrotated_frame.empty())
    {
        printf("ERROR: Unable to grab from the camera \n");
        return -1;
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
    cv::reduce(mask_image, mask_image_summed, 0, 0, CV_32F);
    mask_image_summed.convertTo(mask_image_summed, CV_32S);
    int max_loc[3];
    double max_value;
    cv::minMaxIdx(mask_image_summed,0,&max_value,0,max_loc);
    // Checking that the sum of the columns is correct
    cout << "M = " << endl << " " << mask_image_summed << endl << endl;
    printf("max column %d \n",max_loc[1]);
    printf("max value %f \n",max_value);


    //printf("Channels %d row %d columns%d \n",mask_image_summed.channels(),mask_image_summed.rows,mask_image_summed.cols);
    //cout << "M = " << endl << " " << mask_image_summed.at<int16_t>(10)<< endl << endl;

    // Transform mat to vector
    std::vector<uint16_t> array(224);
    for(int i=0; i<224; i++)
        array.push_back(mask_image_summed.at<int16_t>(i));


    // Find indeces of max_value and then do the mean.
    int counter = 0;
    int index_sum = 0;
    printf("%d \n", array.size());
    for(uint16_t i=0; i<224*2;i++) // old style, no iterator
    {
        if(mask_image_summed.at<int16_t>(i)==(int16_t)max_value)
        {
            index_sum += (i)/2;
            counter += 1;
        }
    }

    int averaged_max_column = (int)(index_sum/counter);
    printf("max averaged column %d \n", averaged_max_column);

    namedWindow( "Mask", WINDOW_AUTOSIZE );
    imshow( "Mask", mask_image );
    namedWindow( "Original", WINDOW_AUTOSIZE );
    imshow( "Original", frame);

    Mat new_mask(224, 224, CV_32FC3, Scalar(0,0,0));
        for(int iheight=0;iheight<224;iheight++)
    {
        for(int iwidth=0;iwidth<224;iwidth++)
        {
            new_mask.at<Vec3f>(iheight,iwidth)[2] = 255.0 * mask_image.at<float>(iheight,iwidth);
            if(iwidth==averaged_max_column)
            {
                new_mask.at<Vec3f>(iheight,iwidth)[1] = 255.0;
            }
        }
    }

    double alpha = 0.8;
    double beta = ( 1.0 - alpha );
    Mat final_mask;
    addWeighted( frame, alpha, new_mask, beta, 0.0, final_mask);

    namedWindow( "Blended", WINDOW_AUTOSIZE );
    imshow( "Blended", final_mask);
    waitKey(0);                                          // Wait for a keystroke in the window


    return 0;
}
