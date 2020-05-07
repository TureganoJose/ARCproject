from __future__ import print_function

import picamera
import picamera.array
import numpy as np


class AnalyseOutput(picamera.array.PiRGBArray):
    def write(self, b):
        result = super(AnalyseOutput, self).write(b)
        self.flush()
        self.analyse(self.array)
        self.buffer = b''
        return result

    def flush(self):
        # Ignore flush when buffer is empty (as it will be when
        # the output is closed)
        if self.buffer:
            super(AnalyseOutput, self).flush()

    def analyse(self, a):
        # Do something with the numpy array here for analysis
        # As an example, we'll calculate the maximum luminance
        # value:
        print(a[..., 0].max())


with picamera.PiCamera() as camera:
    camera.resolution = (1280, 720)
    camera.framerate = 24
    # Start the high-res recording
    camera.start_recording('output.h264')
    # Start the low-res recording to the custom output
    camera.start_recording(
            AnalyseOutput(camera, size=(320, 180)),
            'yuv', resize=(320, 180), splitter_port=2)
    camera.wait_recording(10)
    camera.stop_recording()
    camera.stop_recording(splitter_port=2)