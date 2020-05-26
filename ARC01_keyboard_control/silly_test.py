import time
import picamera

import glob, os, shutil

files = glob.iglob(os.path.join('/home/pi/Documents/ARCproject/ARC02_PS3_js/ARC02_PS3_js/bin/Release', "*.jpg"))
for file in files:
    new_file = file.replace(":","_")
    os.rename(file, new_file)
    if os.path.isfile(file):
        shutil.copy2(new_file, '/media/pi/UUI/New_Camera_Position')
        os.remove(file)
 
# camera = picamera.PiCamera()
# camera.rotation=180
# camera.resolution = (1024, 768)
# camera.start_preview()
# time.sleep(300)
# camera.stop_preview()