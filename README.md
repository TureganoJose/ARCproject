# Sandbox for Autonomous Radio Controlled project
## ARC01 is a cheap RC with a couple of brushed motors to control steering and acceleration.
## ARC02 is based around FTX Vantage chassis platform. It's got a brushed motor and a 3kg servo.

camera_test.py: Contains capturing and saving data benchmarks

bluedot_test.py: Testing the bluedot API. Bluedot is an android app which connects with RPi via Bluetooth and sends the position of your finger placed on a bluedot on the screen. Bluedot also recognises some basic interactions like double-tap and swipe 

Control_and_data_logging.py: It starts 2 threads in parallel the main one. Thread 1 starts the camera capturing. Thread 2 saves the captures as an brg array in the SD. The main thread controls the motor and servo.




Lessons learned:
- How computationally limited a RPi is, as today, an NVidia jetson sounds like the only plausible option beyond a simple end-to-end driving with a CNN.There is also limitations due to camera bandwitdth and memory.
- Overcoming the (computational) limitations of RPI has proved to be a challenge. Now I have a better understanding of multithreadidng (race conditions, inter-locking).
- Project management, how quickly (and easily) the budget can escalate. I've suffered myself planning fallacy as described in "Thinking, fast and slow" from Daniel Kahneman. The bottom line is Don't let optimism bias take over, based estimations on data. In this case the timeframe was half a year, it took over a year. Regarding the cost, the inital budget was around 300 pounds. It just went over 500.
- Coding: definitely helped to improve my coding skills, I do have a better knowledge of python and how to optimise the code to handle large memory chunks. I'm a C++ guy tho. 
 

