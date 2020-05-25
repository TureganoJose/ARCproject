# Sandbox for Autonomous Radio Controlled project


#The project has 5 distinctive parts:
## 1. Prototyping the car
The project started with **ARC01**, a cheap RC with a couple of brushed motors to control steering and acceleration. It highlighted the limitations of the Raspberry pi when it comes to controls  and computational power. 
![ARC01 Bread board](Pictures\ARC01_testing.gif)

## 2. Hardware
The final hardware was based around FTX Vantage chassis platform named **ARC02**. It's got a brushed motor and a 3kg servo, both controlled with PWM through an electronic speed control (ESC). A Raspberry Pi 4 does all the computing. Benchmark suggest its computational power is similar to a low end mobile in 2019 so the neural networks will be designed accordingly.
A chinese replica of Pi Camera v2 was used with very decent results.
Apart from the car, some of the coding and neural network training happened in an old i5-6300HQ (2.3GHz) laptop with a GeForce GTX 960M.

## 3. Controls and software
This was a rather interesting phase of the project. I implemented two different ways of controlling the car:
1. Blue dot app: the RC car can be controlled using an android app which connects with the Raspberry pi via Bluetooth. You are limited to what you can do with this application. In order to start the car, you tap the blue dot. Once the dot has turned green, the car should start (sometimes you need to increase the PWM value depending on the battery charge) and you can steer moving your finger around the dot. Double-tap the blue dot and it will turn red, stopping the car and copying all the pictures taken to an USB.
![Blue dot app](Pictures\bluedotandroid_small.png)
2. PS3 controller, a lot functionality. See picture below.
![PS3 controls, Oxford](Pictures\PS3.png)


## 4. Collecting data and labelling
All pictures and videos were taken different times of the day with different light conditions at University Parks in Oxford.
![University Parks, Oxford](Pictures\Uni_parks.png)
The raw pictures were taken at 1024x768. Due to limitations with SD writing speeds, only 11 pictures are taken per second.
The camera was placed in different places, mostly looking forward at different heights.
Labelme (https://github.com/wkentaro/labelme/blob/master/README.md) was used to label the pictures manually.

## 5. Training models
Two different approaches:
1. Dave-2 end-to-end driving. Everytime a picture was taken, the steering angle applied was logged. Then images (+ augmentation) was fed into the network along with the steering angles. The network then replicates the driving.
Although impressive there is nothing innovative here, it's been done by many people See architecture below, it contains 250 thousand parameters.
![Dave 2 Net](Pictures\Dave_2.png)
2. Using semantic segmentation to detect the road/path and then steer the car accordingly, trying to keep the centerline of the vehicle aligned with the horizon.


The raw pictures were resized to match the netwowrk and augmented with cropping, rotation and gaussian filters.


3 different models were tested for the segmentation approach. 
1. Vainilla Segmentation as specified in https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html. Only 400K parameters. Very simple and decent enough results but good enough. Struggling a lot with some shadows, sky, rocks and sunshine.
![Simple net for Segmentation](Pictures\Simple_Segmentation.png)
2. Unet: typical example of segmentation in medicine to detect cancer but too many parameters (almost 8 million).
![Unet](Pictures\Unet.png)
3. HairMatteNet (https://arxiv.org/pdf/1712.07168.pdf): As the name indicates, originally used to detect hairline. Lightweight segmentation based on MobileNet with a custom decoder (some skip connections and simplified reverse MobileNet). Best results so far, it contains around 4 million parameters but it doesn't mistake benches, sky and sun reflections as the park path. Only 3.8 million
![Simple Segmentation](Pictures\HairMatteNet.png)


Below there are some examples of challenging segmentation with dry patches on the grass, shoes, shadows and reflections. 
![HairMatteNet: Sun flares](Pictures\HairMatteNet_1.png)
![HairMatteNet: bifurcation](Pictures\HairMatteNet_2.png)
![HairMatteNet: bench](Pictures\HairMatteNet_3.png)
![HairMatteNet: shadows](Pictures\HairMatteNet_4.png)
![HairMatteNet: Sun flares](Pictures\HairMatteNet_5.png)
![HairMatteNet: shoes](Pictures\HairMatteNet_6.png)


## 6. Embedded software



camera_test.py: Contains capturing and saving data benchmarks

bluedot_test.py: Testing the bluedot API. Bluedot is an android app which connects with RPi via Bluetooth and sends the position of your finger placed on a bluedot on the screen. Bluedot also recognises some basic interactions like double-tap and swipe 

Control_and_data_logging.py: It starts 2 threads in parallel the main one. Thread 1 starts the camera capturing. Thread 2 saves the captures as an brg array in the SD. The main thread controls the motor and servo.




Lessons learned:
- Got a clearer picture of motors and how they are controlled. For this project, it has been mostly through a ESC (Electronic Speed Controller driven by PWM, sue me I don't have a real-time machine, or the budget for it) but also created my own circuit on a breadbord with a L293D. 
- How computationally limited a RPi is, as of today, an NVidia Jetson sounds like the only plausible option beyond a simple end-to-end driving with a CNN.There is also limitations due to camera bandwitdth and memory.
- Overcoming the (computational) limitations of RPI has proved to be a challenge. Now I have a better understanding of multithreadidng (race conditions, inter-locking) in both environments, Python and C++(OpenMP, good resource: https://csinparallel.org/csinparallel/ppps/openmp.html ).
- Project management, how quickly (and easily) the budget can escalate. I've suffered myself planning fallacy as described in "Thinking, fast and slow" from Daniel Kahneman. The bottom line is Don't let optimism bias take over, based estimations on data. In this case the timeframe was half a year, it took over a year. Regarding the cost, the inital budget was around 300 pounds. It just went over 500.
- Coding: definitely helped to improve my coding skills, I do have a better knowledge of python and how to optimise the code to handle large memory chunks. I'm a C++ guy tho. 
- I've gained some fluency on OpenCV to manipulate images/video and definitely I'm getting better with Keras although I think everyone taking ML seriously should know how to write a neural network framework (to implement custom layers at least)... but that can be found in a different repository.
- Semantic segementation and implementation of more complex architectures 


Notes:
- The code has been developed in different platforms, mostly Ubuntu, Raspbian and Windows 10. Apologies if folders don't work well.

PS3 controller instructions (for Raspberry pi 4):
1. Download sixpair.c from http://pabr.org/sixlinux/sixlinux.en.html
2. Compile it gcc -o sixpair sixpair.c -lusb
3. Connect PS3 to RPi via USB and run ```./sixpair```
4. Write down the bd_addr
5. ```sudo bluetoothctl``` ```agent on``` ```scan on```
6. Press PS button
7. ```connect bd_addr```
8. ```trust bd_addr```

Open MP for Linux:
1. Check your compiler version ```gcc --version```. Anything above 4.2 is compatible I believe.
2. Add the flag ```-fopenmp``` and link to gomp library when compiling
