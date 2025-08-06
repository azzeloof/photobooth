import threading
import time
from gbcamera import GBCamera
from gbprinter import GBPrinter
from nikon import Nikon
from ds40 import DS40
import cv2
import numpy as np


def addGameboyBorder(frame):
    assert frame.shape[1] == 128 and frame.shape[0] == 112, "Frame is not 128x112"
    gb_border = cv2.imread('gb_border.png')
    gb_border[16:16+frame.shape[0], 16:16+frame.shape[1]] = frame
    return gb_border


def gbColorize(frame):
    colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Define the color mapping for each grayscale value
    color_map = {
        0: [41, 65, 57],
        96: [57, 89, 74],
        178: [90, 121, 66],
        255: [123, 130, 16]
    }
    for gray_val, color_val in color_map.items():
        colorFrame[np.all(frame == [gray_val, gray_val, gray_val], axis=-1)] = color_val
    return frame


class Photobooth:
    def __init__(self):
        self.gbCamera = GBCamera()
        self.gbPrinter = GBPrinter()
        self.nikon = Nikon()
        self.ds40 = DS40()
        self.running = True
        self.gbThread = threading.Thread(target=self.gbCamera.run)
        self.gbThread.start()

    def run(self):
        while self.running:
            gbFrame = self.gbCamera.getFrame()
            gbFrameColor = gbColorize(gbFrame)
            if gbFrame is not None:
                cv2.imshow('Camera', cv2.resize(gbFrameColor, (800, 700)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                if gbFrame is not None:
                    timestamp = int(time.time())
                    cv2.imwrite(f'captures/frame_{timestamp}.png', gbFrame)
                    cv2.imwrite(f'captures/frameBordered_{timestamp}.png', addGameboyBorder(gbFrame))
                    self.gbPrinter.printFrame(gbFrame)
                    print("frame captured")
        self.shutdown()

    def shutdown(self):
        self.gbCamera.stop()
        self.gbThread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    photobooth = Photobooth()
    photobooth.run()