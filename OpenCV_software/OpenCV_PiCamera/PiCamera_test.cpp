
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ctime>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc,char ** argv)
{
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    cerr << "ERROR: Unable to open the camera" << endl;
    return 0;
  }

  int icounter = 0;
  Mat frame;
  cout << "Start grabbing, press a key on Live window to terminate" << endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  while(1) {
    cap >> frame;
    if (frame.empty()) {
        cerr << "ERROR: Unable to grab from the camera" << endl;
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
        std::string str(buffer);
        result = imwrite("Tick"+ctr_str+"Data"+str+".jpg", frame);
    }
    catch (const cv::Exception& ex)
    {
        fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
    }
    if (result)
        icounter += 1;
        //printf("Saved JPG file with alpha data.\n");
    else
        printf("ERROR: Can't save JPG file.\n");

    // Show video
//    imshow("Live",frame);
    int key = cv::waitKey(20);
    key = (key==255) ? -1 : key;
    if (key>=0)
      break;
      if(icounter==500)
        break;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference = " << std::chrono::duration_cast <std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
  printf("%d captures \n",icounter);

  cout << "Closing the camera" << endl;
  cap.release();
  destroyAllWindows();
  cout << "bye!" <<endl;
  return 0;
}
