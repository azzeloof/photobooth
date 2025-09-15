import cv2
import numpy as np
import time

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

    def addBorder(self, frame):
        assert frame.shape[1] == 128 and frame.shape[0] == 112, "Frame is not 128x112"
        gb_border = cv2.imread('gb_border.png')
        gb_border[16:16 + frame.shape[0], 16:16 + frame.shape[1]] = frame
        return gb_border

    def colorize(frame):
        if frame is None:
            return frame
        x, y, c = frame.shape
        colorFrame = np.zeros((x, y, 3), dtype=np.uint8)
        # Define the color mapping for each grayscale value
        color_map = {
            0: [41, 65, 57],
            96: [57, 89, 74],
            178: [90, 121, 66],
            255: [123, 130, 16]
        }
        for gray_val, color_val in color_map.items():
            colorFrame[np.all(frame == [gray_val, gray_val, gray_val], axis=-1)] = color_val
        return colorFrame

    def getFrame(self, color=False):
        if color:
            return self.colorize(self.currentFrame)
        else:
            return self.currentFrame

    def capture(self):
        timestamp = int(time.time())
        frame = self.getFrame()
        frameFile = str(f'captures/frame_{timestamp}.png')
        frameBorderedFile= str(f'captures/frameBordered_{timestamp}.png')
        cv2.imwrite(frameFile, frame)
        cv2.imwrite(frameBorderedFile, self.addBorder(frame))
        return (frameFile, frameBorderedFile)

    def stop(self):
        self.running = False
        self.cam.release()
