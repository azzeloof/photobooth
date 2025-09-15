import threading
from sys import exception

from gbcamera import GBCamera
from gbprinter import GBPrinter
from nikon import Nikon
from hardware import Hardware
from ds40 import DS40
import cv2
import time


def layoutPage(frames):
    pass


class PhotoSet:
    def __init__(self, n):
        self.time = time.time()
        self.nCaptured = 0
        self.slots = n
        self.captures = []

    def addPhoto(self, photo):
        if self.n_captures() < self.slots:
            self.captures.append(photo)
            return 0
        return 1

    def n_captures(self):
        return len(self.captures)

    def save(self):
        pass



class Photobooth:
    def __init__(self):
        #################### Parameters ####################
        self.nCaptures = 3      # Number of frames to capture on each capture cycle
        self.captureTimeout = 5 # Seconds between captures
        ####################################################

        # Hardware Interfaces
        self.gbCamera = GBCamera()
        self.gbPrinter = GBPrinter()
        self.nikon = Nikon()
        self.ds40 = DS40()
        try:
            self.hw = Hardware()
        except:
            print("no hw")
            self.hw = None
        self.windowName = "Photobooth"
        if self.hw != None:
            self.hw.registerCallback(0, self.captureAndPrint)
        cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Flags
        self.running = True
        self.capturing = False
        self.captureFlag = False
        self.printFlag = False
        self.captureTimer = 0 # seconds, countdown to a photo being taken

        # Initialization
        self.gbThread = threading.Thread(target=self.gbCamera.run)
        if self.hw is not None:
            self.hwThread = threading.Thread(target=self.hw.run)
        self.gbThread.start()
        if self.hw is not None:
            self.hwThread.start()


    def run(self):
        recentFrames = (None, None)
        while self.running:
            gbFrame = self.gbCamera.getFrame(color=True)
            if gbFrame is not None:
                # Display something on the monitor
                cv2.imshow(self.windowName, cv2.resize(gbFrame, (0, 0), fx=4, fy=4))
                #TODO: Display countdown during a capture
            key = cv2.waitKey(1) & 0xFF
            if self.captureFlag:
                gbFile, gbFileBordered = self.gbCamera.capture()
                nikonFile = self.nikon.capture()
                recentFrames = (gbFile, nikonFile)
                print("frame captured")
                self.captureFlag = False
            if self.printFlag:
                # print
                page = layoutPage(recentFrames)
                self.ds40.print(page)
                self.printFlag = False
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self.capture()
        self.shutdown()

    def capture(self):
        """
        Initiates a capture process
        """
        self.captureFlag = True

    def print(self):
        """
        Initiates a print process
        """
        self.printFlag = True

    def captureAndPrint(self):
        """
        Kicks off a capture and print process
        """
        self.capture()
        self.print()

    def shutdown(self):
        """
        Shuts down all the things
        """
        self.gbCamera.stop()
        self.gbThread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    photobooth = Photobooth()
    photobooth.run()
