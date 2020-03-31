#https://javatutorial.net/raspberry-pi-control-dc-motor-speed-and-direction-java

import RPi.GPIO as GPIO
from time import sleep
import time
#from pynput.mouse import Listener
from pynput.keyboard import Key, Listener

# Pins for Motor Driver Inputs
Motor1A = 24
Motor1B = 23
Motor1E = 25

Motor2A = 27
Motor2B = 17
Motor2E = 22

# The getch method can determine which key has been pressed
# by the user on the keyboard by accessing the system files
# It will then return the pressed key as a variable
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def setup():
    GPIO.setmode(GPIO.BCM) # GPIO Numbering
    GPIO.setup(Motor1A,GPIO.OUT) # All pins as Outputs
    GPIO.setup(Motor1B,GPIO.OUT)
    GPIO.setup(Motor1E,GPIO.OUT)
    GPIO.setup(Motor2A,GPIO.OUT) # All pins as Outputs
    GPIO.setup(Motor2B,GPIO.OUT)
    GPIO.setup(Motor2E,GPIO.OUT)

def motor_forwards():
    # Going forwards
    GPIO.output(Motor1A,GPIO.HIGH)
    GPIO.output(Motor1B,GPIO.LOW)
    GPIO.output(Motor1E,GPIO.HIGH)
    
def motor_backwards():
    # Going backwards
    GPIO.output(Motor1A,GPIO.LOW)
    GPIO.output(Motor1B,GPIO.HIGH)
    GPIO.output(Motor1E,GPIO.HIGH)

def motor_right():
    # Steer right
    GPIO.output(Motor2A,GPIO.HIGH)
    GPIO.output(Motor2B,GPIO.LOW)
    GPIO.output(Motor2E,GPIO.HIGH)
    
def motor_left():
    # Steer left
    GPIO.output(Motor2A,GPIO.LOW)
    GPIO.output(Motor2B,GPIO.HIGH)
    GPIO.output(Motor2E,GPIO.HIGH)
    
def stop():
    GPIO.output(Motor1E,GPIO.LOW)

def loop():
    # Going forwards
    GPIO.output(Motor1A,GPIO.HIGH)
    GPIO.output(Motor1B,GPIO.LOW)
    GPIO.output(Motor1E,GPIO.HIGH)

    sleep(5)
    # Going backwards
    GPIO.output(Motor1A,GPIO.LOW)
    GPIO.output(Motor1B,GPIO.HIGH)
    GPIO.output(Motor1E,GPIO.HIGH)

    sleep(5)
    # Stop
    GPIO.output(Motor1E,GPIO.LOW)

def destroy():
    GPIO.cleanup()


def on_click(x, y, button, pressed):
    if pressed:
        print ("Mouse clicked")
        print(button)
        motor_forwards()
    else:
        print("Released")
        stop()

#with Listener(on_click=on_click) as listener:
#    listener.join()
setup()

def on_press(key):
    
    if key.char == ('w'):
        motor_forwards()
    elif key.char == ('s'):
        motor_backwards()
    elif key.char == ('a'):
        motor_left()
    elif key.char == ('d'):
        motor_right()
    
    
    
def on_release(key):
    stop()
    if key == Key.esc:
        destroy()
        #Stop listener
        return False
with Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()
    
    
    
destroy()

# while True:
#     char = getch()
# 
#     if (char == "q"):
#         stop()
#         destroy()
#         exit(0)  
# 
#     if (char == "a"):
#         print('Left pressed')
#         stop()
#         time.sleep(button_delay)
# 
#     if (char == "d"):
#         print('Right pressed')
#         stop()
#         time.sleep(button_delay)          
# 
#     elif (char == "w"):
#         print('Up pressed') 
#         motor_forwards()       
#         time.sleep(button_delay)          
#     
#     elif (char == "s"):
#         print('Down pressed')      
#         motor_backwards()
#         time.sleep(button_delay)  
#     
#     stop()
