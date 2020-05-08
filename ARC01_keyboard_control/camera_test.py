
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import numpy as np
import pickle
import cv2

class PiVideoStream:

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
        
def detect_in_thread():
    # Start updating frames in threaded manner
    thread_stream = PiVideoStream()
    thread_stream.start()
    time.sleep(2)
    
    ## Preallocating memory
    #Array_mem_map = np.memmap('/media/pi/UUI/capture.npy',dtype='float32', mode='w+', shape=(10,768,1024,3))

    
    iCount = 0
    
    # Read frames
    while True:
        start = time.time()
        # Original image
        image = thread_stream.read()
        finish = time.time()
        print('Captured 1 image at %.8fs' % ( (finish - start))) 

        # Moving the vehicle here and saving the image
        
        start = time.time()
        ## Checking size of picture
        #print('Captured %dx%dx%d image' % (
        #        image.shape[1], image.shape[0],image.shape[2]))
        #print('number of bytes %f' % (image.nbytes))
        
        ## Writing in USB with pickle 0.2-0.3s
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #with open('/media/pi/UUI/'+timestr+'Tick'+str(iCount)+'.npy', 'wb') as f:
        #    pickle.dump(image,f)
        
        ## Writing in USB with numpy 0.2-0.3s
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #np.save('/media/pi/UUI/'+timestr+'Tick'+str(iCount)+'.npy',image)
    
        ## Writing to SD card 0.01-0.02s
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #np.save(timestr+'Tick'+str(iCount)+'.npy',image)
    
        ## Writing to SD card 0.01-0.02s
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #output_file = open(timestr+'Tick'+str(iCount)+'.npy', 'wb')
        #image.tofile(output_file)
        #output_file.close()

        ## Writing to SD card with opencv 0.01-0.02s
        timestr = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(timestr+'Tick'+str(iCount)+'.bgr',image)
    
        ## Mapping memory to USB 0.1-0.2s
        #Array_mem = np.memmap('/media/pi/UUI/'+timestr+'Tick'+str(iCount)+'.npy',dtype='float32', mode='w+', shape=(480,640,3))
        #Array_mem[:] = image[:]
        #del Array_mem
        
        ## Preallocating memory: Quickly running out of memory
        #Array_mem_map[iCount,:,:,:]=image[:,:,:]
        
        
        if iCount == 9:
            #del Array_mem_map
            break
        iCount += 1
        finish = time.time()
        print('Saving 1 image at %.8fs' % ( (finish - start))) 
    # Close thread
    thread_stream.stop()


if __name__ == "__main__":
    detect_in_thread()    
       
       
       
       
       
       
       
# import picamera
# import time
# import io
# from picamera.array import PiRGBArray

# with picamera.PiCamera() as camera:
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
    
#     camera.resolution = (1024, 768)
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


#     start = time.time()
#     rawCapture = PiRGBArray(camera, size=(1024, 768))
#     camera.framerate = 80
#     count=0
#     # capture frames from the camera
#     for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#         image = frame.array
# 
#         # clear the stream in preparation for the next frame
#         rawCapture.truncate(0)
#         count +=1
#         # if the `q` key was pressed, break from the loop
#         if count == 10:
#             break
#     finish = time.time()
#     print(count)
#     print('Captured x image at %.2fs' % ( (finish - start))) 