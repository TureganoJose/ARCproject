import time
import picamera
camera = picamera.PiCamera()
camera.rotation=180
camera.resolution = (1024, 768)
camera.start_preview()
time.sleep(60)
camera.stop_preview()