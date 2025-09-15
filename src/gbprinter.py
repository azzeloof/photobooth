# Based (heavily) on https://github.com/Staacks/there.oughta.be/blob/master/game-boy-photo-booth/GBPrinter/GBPrinter.py

import serial
import time
from PIL import Image
import spidev

class GBPrinter:

    debug = False

    def __init__(self, device=None, baud=None):
        #self.device = device
        #self.baud = baud
        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = 1200000

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False

    class PrinterException(Exception):
        pass

    def buildCommand(self, cmd, data):
        magicbytes = b'\x88\x33'
        command = cmd.to_bytes(1)
        compressionflag = b'\x00'
        length = len(data).to_bytes(2, byteorder="little")
        tx = magicbytes + command + compressionflag + length + data
        checksum = (sum(tx[2:]) % 0x10000).to_bytes(2, byteorder="little")
        tx += checksum
        return tx

    def sendCommand(self, cmd, data):
        tx = self.buildCommand(cmd, data) + b'\x00\x00'
        if self.debug:
            print("Sending " + " ".join(f'{c:02x}' for c in tx)) 
        with serial.Serial(self.device, self.baud, timeout=1) as ser:
            ser.write(tx)
            response = ser.read(len(tx)) #IMPORTANT: Always read as many bytes as have been sent. The micro controller reads synchronously from SPI and it is only done sending the command when it has returned the same number of bytes.
            if self.debug:
                print("Received " + " ".join(f'{c:02x}' for c in response))
            if response[-2] != 0x81:
                raise self.PrinterException("Invalid response from Game Boy printer")
            return response[-1]
        raise self.PrinterException("Failed to open serial port") 

    def initialize(self):
        return self.sendCommand(0x01, b'')

    #sheets = 0..255
    #marginBefore = 0..15
    #marginAfter = 0..15
    #palette = 1byte palette definition
    #exposure = 0.0..1.0
    def startPrint(self, sheets, marginBefore, marginAfter, palette, exposure):
        exposurebyte = round(exposure * 0x7f)
        data = sheets.to_bytes(1) + (marginBefore << 4 | marginAfter).to_bytes(1) + palette.to_bytes(1) + exposurebyte.to_bytes(1)
        result = self.sendCommand(0x02, data)
        if result & 0xf1 != 0:
            raise self.PrinterException("Failed to start print. Result: " + f'{result:02x}')

    def fill(self, data):
        i = 0
        while i < len(data):
            result = self.sendCommand(0x04, data[i:i+640])
            if result & 0xf1 != 0:
                raise self.PrinterException("Failed to send data. Result: " + f'{result:02x}')
            i += 640
        result = self.sendCommand(0x04, b'')
        if result & 0xf1 != 0:
            raise self.PrinterException("Failed to send data. Result: " + f'{result:02x}')

    def status(self):
        return self.sendCommand(0x0f, b'')

    def waitForEndOfPrint(self):
        print("Wait for end of print...")
        state = 0x02
        while state & 0x02 != 0:
            time.sleep(0.5)
            state = self.status()
            if state & 0xf1 != 0:
                raise self.PrinterException("Received error while waiting for print to finish: " + f'{state:02x}')
        time.sleep(0.1)
        print("Printer ready.")

    def pixelRowToTiles(self, pixels):
        data = bytearray(20*16) #20 tiles in a row, each tile has 16 bytes representing 8x8 pixels
        if self.debug:
            print("Generating row of tiles")
        for i in range(20): #for each tile
            for y in range(8): #for each line within the tile
                dataoffset = 16*i + 2*y
                for x in range(8): #for each column within the tile
                    pixel = pixels[160*y+8*i+x]
                    if pixel & 0x40 == 0:
                        data[dataoffset] |= (0x80 >> x)
                    if pixel & 0x80 == 0:
                        data[dataoffset + 1] |= (0x80 >> x)
            if self.debug:
                print("Tile " + str(i) + ": " + " ".join(f'{c:02x}' for c in data[16*i:16*(i+1)]))
        return data

    def pixelsToTiles(self, pixels):
        i = 0
        data = b''
        while i < len(pixels):
            data += self.pixelRowToTiles(pixels[i:i+8*160])
            i += 8*160
        return data

    # Pixels is a byte array with each byte representing a grayscale pixel in "modern" representation of 0=black to 255=white
    def printImage(self, pixels, exposure):
        print("Check printer presence and initialize.")
        self.status()
        self.initialize()
        state = self.status()
        if state != 0x00:
            raise self.PrinterException("Unexpected status after initialization: " + f'{state:02x}')

        if len(pixels) % (16*160) != 0:
            raise self.PrinterException("Image height must be a multiple of 16!")
        multiPartPrint = len(pixels) > 144*160

        if multiPartPrint: #Manually add margin to print without margins
            print("Adding empty margin for multi-part print")
            self.initialize()
            self.fill(b'')
            self.startPrint(0, 0, 1, 0b11100100, exposure)
            self.waitForEndOfPrint()

        i = 0
        while i < len(pixels):
            print("Printing...")
            self.initialize()
            self.fill(self.pixelsToTiles(pixels[i:i+144*160]))
            self.startPrint(1, 0 if multiPartPrint else 1, 0 if multiPartPrint else 3, 0b11100100, exposure)
            self.waitForEndOfPrint()
            i += 144*160

        if multiPartPrint: #Manually add margin to print without margins
            print("Adding empty margin for multi-part print")
            self.initialize()
            self.fill(b'')
            self.startPrint(0, 0, 3, 0b11100100, exposure)
            self.waitForEndOfPrint()

    def printImageFromFile(self, path, exposure):
        img = Image.open(path)
        img = img.resize((160, 160*img.size[1]//img.size[0]), Image.Resampling.LANCZOS)
        pixels = img.convert('L').tobytes()
        self.printImage(pixels, exposure)
