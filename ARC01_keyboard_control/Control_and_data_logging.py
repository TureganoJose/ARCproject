from bluedot import BlueDot
from signal import pause
import os
import time
import numpy as np
os.system ("sudo pigpiod") #Launching GPIO library
time.sleep(1) # Delay to launch the GPIO library
import pigpio

from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import numpy as np
import cv2

import glob, os, shutil


class PiVideoStream:
    ''' Class for thread handling the video stream'''

    def __init__(self, resolution=(1024, 768), framerate=20):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,format='bgr', use_video_port=True)
        self.image = None
        self.stopped = False

    def start(self):
        t = Thread(target=self.update)
        t.daemon = True
        t.start()
        return self

    def update(self):
        for frame in self.stream:
            self.image = frame.array
            self.rawCapture.truncate(0)

            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return  

    def read(self):
        return self.image

    def stop(self):
        self.stopped = True

class car_logging():
    ''' Class for thread handling saving captures in SD card'''
    
    def __init__(self):
        # So far only video and steering
        self.steering_angle = 0.0
        self.array = np.zeros((768, 1024, 3))
        self.steering_angle_old = -1000
        self.stopped = False
        
    def start(self):
        # Starting its own thread
        ts = Thread(target=self.save_array)
        ts.daemon = True
        ts.start()
        #ts.join()
        return self

    def save_array(self):
        global count
        while True:
            #print(count)
            #print(self.steering_angle)
            if abs(self.steering_angle - self.steering_angle_old)>0.001: # Instead of using events/lock
                count +=1
                timestr = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(timestr+'Tick'+str(count)+'str'+str(self.steering_angle)+'.jpg',self.array)
                #np.save(timestr+'Tick'+str(1)+'str'+str(self.steering_angle)+'.npy',self.array)
                self.steering_angle_old = self.steering_angle
            if self.stopped:
                break

    def update_logging(self, image, steering):
        self.steering_angle = steering
        self.array = image

    def stop(self):
        self.stopped = True
        
        
def start_stop():
    global toggle
    global speed
    if toggle == 1:
        bd.color = 'green'
        toggle = 0
        # Start threads
        thread_stream.start()
        thread_carlog.start()
        motor_speed_start(speed)
    else:
        stop()
        bd.color = (255, 165, 0)#orange
        thread_stream.stop()
        thread_carlog.stop()
        copy_and_images()
        pi.stop()
        bd.color = 'red'
        toggle = 1

def steering(pos):
    # Positive right, negative left
    #global count 
    steering_servo(pos.angle)
    # capture stream
    image = thread_stream.read()
    # save image matrix
    thread_carlog.update_logging(image, pos.angle)
    
def motor_speed_start(speed):
    speed = np.clip(speed, min_esc_value, max_esc_value)
    pi.set_servo_pulsewidth(ESC, speed)
    #print('speed',speed)
    
def stop_motor(): #stop ESC motor
    pi.set_servo_pulsewidth(ESC, 0)
    #print('stopping motor')
    
def stop_servo(): #stop servo
    pi.set_servo_pulsewidth(servo, 0)
    
def steering_servo(angle): # steering servo
    steering =  zero_value - half_range * angle/180 # angle goes from -180 to 180
    steering = np.clip(steering, min_servo_value, max_servo_value)
    pi.set_servo_pulsewidth(servo, steering)

def stop():
    stop_motor()
    stop_servo()

def copy_and_images():
    '''Copy captures and move them to an external drive'''
    files = glob.iglob(os.path.join('/home/pi/Documents/ARCproject/ARC01_keyboard_control', "*.jpg"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, '/media/pi/UUI')
            os.remove(file)


#Initial parameters

ESC = 4 # GPIO 4 (pin 7)
servo = 17 # GPIO 17 (pin 11)
bd = BlueDot()
count = 0
toggle = 1
speed = 1560.0

pi = pigpio.pi();
pi.set_servo_pulsewidth(ESC, 0) 
pi.set_servo_pulsewidth(servo, 0)

max_servo_value = 1780 #Servo's max value
min_servo_value = 980  #Servo's min value

zero_value = (max_servo_value + min_servo_value)/2
half_range = (max_servo_value - min_servo_value)/2
    
max_esc_value = 2000 #ESC's max value
min_esc_value = 980  #ESC's min value

increase_throttle_rate = 10
decrease_throttle_rate = 10

# Video stream in one thread
thread_stream = PiVideoStream()
thread_carlog = car_logging()
time.sleep(2)

bd.rotation_segments = 180
bd.color = 'red'

bd.when_double_pressed = start_stop
bd.when_moved = steering

pause()