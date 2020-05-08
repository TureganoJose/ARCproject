import cv2
import numpy as np

#image = np.load('/media/pi/UUI/20200508-162114Tick1str-58.58604083039191.npy')
image=cv2.imread('/media/pi/UUI/download.jpeg')
print(type(image))
print(image.shape)
#cv2.imshow('test', image)