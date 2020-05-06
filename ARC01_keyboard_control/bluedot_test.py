from bluedot import BlueDot
from signal import pause
import os
import time
import numpy as np
os.system ("sudo pigpiod") #Launching GPIO library
time.sleep(1) # Delay to launch the GPIO library
import pigpio 
from picamera import PiCamera

camera = PiCamera()
camera.rotation = 180
camera.resolution = (1024, 768)



ESC = 4 # GPIO 4 (pin 7)
servo = 17 # GPIO 17 (pin 11)
bd = BlueDot()
count = 0.0
toggle = 1
speed = 1645.0

pi = pigpio.pi();
pi.set_servo_pulsewidth(ESC, 0) 
pi.set_servo_pulsewidth(servo, 0)

max_servo_value = 1780 #Servo's max value
min_servo_value = 980  #Servo's min value

zero_value = (max_servo_value + min_servo_value)/2
half_range = (max_servo_value - min_servo_value)/2
    
max_esc_value = 1780 #ESC's max value
min_esc_value = 980  #ESC's min value

increase_throttle_rate = 10
decrease_throttle_rate = 10


## Blue dot function to test
def move(pos):
    if pos.top:
        update_motor_speed(pos.y*increase_throttle_rate)
        #print('increasing motor speed by',pos.y*increase_throttle_rate)
    elif pos.bottom:
        update_motor_speed(pos.y*decrease_throttle_rate)
        #print('decreasing motor speed by',pos.y*decrease_throttle_rate)
#     elif pos.left:
#         print('left',pos.distance)
#     elif pos.right:
#         print('right',pos.distance)

def stop():
    stop_motor()
    stop_servo()
    #print('stop')

def start_stop():
    global toggle
    global speed
    if toggle == 1:
        #print('starting')
        bd.color = 'green'
        toggle = 0
        motor_speed_start(speed)
        #camera.start_preview()
    else:
        #print('stoping')
        stop()
        bd.color = 'red'
        toggle = 1
        #camera.stop_preview()

def steering_ipod(angle):
    # Positive right, negative left
    global count
    count += angle.value
    steering_servo(count)
    #print('steering',count)

def steering(pos):
    # Positive right, negative left
    #print('angle',pos.angle)
    global count
    count += 1.0
    print(count)
    steering_servo(pos.angle)
    
## ESC and servo functions
def update_motor_speed(speed_update):
    global speed
    speed += speed_update
    speed = np.clip(speed, min_esc_value, max_esc_value)
    #pi.set_servo_pulsewidth(ESC, speed)
    #print('increasing speed',speed)
    
def motor_speed_start(speed):
    speed = np.clip(speed, min_esc_value, max_esc_value)
    pi.set_servo_pulsewidth(ESC, speed)
    #print('speed',speed)
    
def stop_motor(): #stop ESC motor
    pi.set_servo_pulsewidth(ESC, 0)
    #print('stopping motor')
    
def stop_servo(): #stop servo
    pi.set_servo_pulsewidth(servo, 0)
    #print('stopping servo')

def steering_servo(angle): # steering servo
    steering =  zero_value - half_range * angle/180 # angle goes from -180 to 180
    steering = np.clip(steering, min_servo_value, max_servo_value)
    pi.set_servo_pulsewidth(servo, steering)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #camera.capture('/media/pi/UUI/'+timestr+'str'+str(steering)+'.jpg')
    #print('steering', steering)
    print('/media/pi/UUI/'+timestr+'str'+str(steering)+'.jpg')

bd.rotation_segments = 180
bd.color = 'red'


#bd.when_pressed = update_motor_speed
#bd.when_moved = move
#bd.when_released = stop
bd.when_double_pressed = start_stop
#bd.when_rotated = steering_ipod
bd.when_moved = steering
print('hola')
pause()