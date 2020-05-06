import picamera
import time
import io
from picamera.array import PiRGBArray

with picamera.PiCamera() as camera:
#     camera.rotation = 180
#     camera.start_preview()
#     time.sleep(5)
#     camera.stop_preview()
#     camera.resolution = (1024, 768)
#     camera.start_preview()
#     # Camera warm-up time
#     time.sleep(2)
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     camera.capture('/media/pi/UUI/'+timestr+'.jpg')
#     print(timestr)
    
    camera.resolution = (1024, 768)
#     camera.start_recording('/media/pi/UUI/my_video.h264')
#     camera.wait_recording(60)
#     camera.stop_recording()
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     print(timestr)



    ## stream
#     camera.framerate = 80
#     time.sleep(1)
#     start = time.time()
#     # Set up 40 in-memory streams
#     for x in range(10):
#         outputs = io.BytesIO(40)
#         timestr = time.strftime("%Y%m%d-%H%M%S")
#         camera.capture(outputs, format='bgr', use_video_port=True)
#     finish = time.time()
#         # How fast were we?
#     print('Captured 1 image at %.2fs' % ( (finish - start))) 


    start = time.time()
    rawCapture = PiRGBArray(camera, size=(1024, 768))
    camera.framerate = 80
    count=0
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        count +=1
        # if the `q` key was pressed, break from the loop
        if count == 10:
            break
    finish = time.time()
    print(count)
    print('Captured x image at %.2fs' % ( (finish - start))) 