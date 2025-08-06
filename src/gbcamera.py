import cv2
import numpy as np

class GBCamera:
    def __init__(self, device=0):
        self.cam = cv2.VideoCapture(device)
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.currentFrame = None
    
    def crop(self, frame):
        start_x = 128
        start_y = 128
        fx = 128
        fy = 112
        scale = 8
        end_x = int(start_x + fx*scale)
        end_y = int(start_y + fy*scale)
        frame = frame[start_y:end_y, start_x:end_x]
        frame = cv2.resize(frame, (fx, fy), interpolation=cv2.INTER_AREA)
        return frame

    def quantize(self, frame):
        frame = np.where((frame >= 0) & (frame <= 64), 0, frame)
        frame = np.where((frame >= 65) & (frame <= 128), 96, frame)
        frame = np.where((frame >= 129) & (frame <= 192), 178, frame)
        frame = np.where((frame >= 193) & (frame <= 255), 255, frame)
        return frame
    
    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cam.read()
            try:
                frame = self.crop(frame)
                frame = self.quantize(frame)
            except:
                print("frame processing error")
            self.currentFrame = frame

    def getFrame(self):
        return self.currentFrame

    def stop(self):
        self.running = False
        self.cam.release()