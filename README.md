# Sandbox for Autonomous Radio Controlled project
## ARC01 is a cheap RC with a couple of brushed motors to control steering and acceleration.
## ARC02 is based around FTX Vantage chassis platform. It's got a brushed motor and a 3kg servo.



#The project has 5 distinctive parts:
## 1. Prototyping the car
The project started with **ARC01**, a cheap RC with a couple of brushed motors to control steering and acceleration. It highlighted the limitations of the raspberry pi when it comes to controls  and computational power. 
## 2. Hardware
## 3. Controls and software
## 4. Collecting data and labelling
## 5. Training models
## 6. Embedded software



camera_test.py: Contains capturing and saving data benchmarks

bluedot_test.py: Testing the bluedot API. Bluedot is an android app which connects with RPi via Bluetooth and sends the position of your finger placed on a bluedot on the screen. Bluedot also recognises some basic interactions like double-tap and swipe 

Control_and_data_logging.py: It starts 2 threads in parallel the main one. Thread 1 starts the camera capturing. Thread 2 saves the captures as an brg array in the SD. The main thread controls the motor and servo.


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



Lessons learned:
- Got a clearer picture of motors and how they are controlled. For this project, it has been mostly through a ESC (Electronic Speed Controller driven by PWM, sue me I don't have a real-time machine, or the budget for it) but also created my own circuit on a breadbord with a L293D. 
- How computationally limited a RPi is, as of today, an NVidia Jetson sounds like the only plausible option beyond a simple end-to-end driving with a CNN.There is also limitations due to camera bandwitdth and memory.
- Overcoming the (computational) limitations of RPI has proved to be a challenge. Now I have a better understanding of multithreadidng (race conditions, inter-locking) in both environments, Python and C++(OpenMP, good resource: https://csinparallel.org/csinparallel/ppps/openmp.html ).
- Project management, how quickly (and easily) the budget can escalate. I've suffered myself planning fallacy as described in "Thinking, fast and slow" from Daniel Kahneman. The bottom line is Don't let optimism bias take over, based estimations on data. In this case the timeframe was half a year, it took over a year. Regarding the cost, the inital budget was around 300 pounds. It just went over 500.
- Coding: definitely helped to improve my coding skills, I do have a better knowledge of python and how to optimise the code to handle large memory chunks. I'm a C++ guy tho. 
 

