import board
import digitalio
import time
from adafruit_seesaw.seesaw import Seesaw
from adafruit_seesaw.digitalio import DigitalIO
from adafruit_seesaw.pwmout import PWMOut

class Button:
    def __init__(self, controller, buttonPin, ledPin, cb=None):
        self.controller = controller
        self.buttonPin = buttonPin
        self.ledPin = ledPin
        self.cb = cb
        self.btn = DigitalIO(self.controller, self.buttonPin)
        self.btn.direction = digitalio.Direction.INPUT
        self.btn.pull = digitalio.Pull.UP
        self.led = PWMOut(self.controller, self.ledPin)
        self.state = False
        self.prevState = True

    def registerCallback(self, cb):
        self.cb = cb

    def poll(self):
        if self.btn.value:
            self.state = True
            if self.prevState == False:
                # The button has just been pressed
                if self.cb != None:
                    self.cb()
        else:
            self.state = False
        self.prevState = self.state


class Hardware:
    def __init__(self):
        #self.spi = spidev.SpiDev()
        self.i2c = board.I2C()
        self.seesaw = Seesaw(self.i2c, addr=0x3A)
        # button pin : led pin
        # ordered dict, requires 3.7 or newer
        self.pins = {
            18: 12,
            19: 13,
            20: 0,
            2: 1
        }
        self.buttons = []
        for buttonPin, ledPin in self.pins.items():
            self.buttons.append(Button(
                self.seesaw,
                buttonPin,
                ledPin,
                None
            ))
        self.running = False

    def registerCallback(self, button, callback):
        self.buttons[button].registerCallback(callback)

    def run(self):
        self.running = True
        while self.running:
            for button in self.buttons:
                button.poll()
            time.sleep(0.01)    

    def stop(self):
        self.running = False
        


if __name__ == "__main__":
    hw = Hardware()
    hw.registerCallback(0, lambda: print("ayyyyyyyyyy"))
    hw.run()