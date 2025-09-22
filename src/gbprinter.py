# Based (heavily) on https://github.com/Staacks/there.oughta.be/blob/master/game-boy-photo-booth/GBPrinter/GBPrinter.py

import time
from PIL import Image
import spidev


class GBPrinter:
    debug = False

    def __init__(self, bus=0, device=0):
        """Initializes the printer using a specific SPI bus and device."""
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = 100000  # You can adjust this speed if needed
        #self.spi.mode = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Closes the SPI connection upon exiting the 'with' block."""
        self.spi.close()
        return False

    class PrinterException(Exception):
        pass

    def buildCommand(self, cmd, data):
        magicbytes = b'\x88\x33'
        command = cmd.to_bytes(1, 'big')
        compressionflag = b'\x00'
        length = len(data).to_bytes(2, byteorder="little")
        tx = magicbytes + command + compressionflag + length + data
        checksum = (sum(tx[2:]) % 0x10000).to_bytes(2, byteorder="little")
        tx += checksum
        return tx

    def sendCommand(self, cmd, data):
        """Builds and sends a command over SPI, then returns the printer's status byte."""
        tx = self.buildCommand(cmd, data) + b'\x00\x00'
        if self.debug:
            print("Sending " + " ".join(f'{c:02x}' for c in tx))

        # Use spidev's xfer2 method for synchronous transfer
        response_list = self.spi.xfer2(list(tx))
        response = bytes(response_list)

        if self.debug:
            print("Received " + " ".join(f'{c:02x}' for c in response))

        if len(response) < 2 or response[-2] != 0x81:
            raise self.PrinterException(
                "Invalid response from Game Boy printer. Received: " + " ".join(f'{c:02x}' for c in response))

        return response[-1]

    def initialize(self):
        return self.sendCommand(0x01, b'')

    def startPrint(self, sheets, marginBefore, marginAfter, palette, exposure):
        exposurebyte = round(exposure * 0x7f)
        data = (
                sheets.to_bytes(1, 'big') +
                (marginBefore << 4 | marginAfter).to_bytes(1, 'big') +
                palette.to_bytes(1, 'big') +
                exposurebyte.to_bytes(1, 'big')
        )
        result = self.sendCommand(0x02, data)
        if result & 0xf1 != 0:
            raise self.PrinterException("Failed to start print. Result: " + f'{result:02x}')

    def fill(self, data):
        i = 0
        while i < len(data):
            result = self.sendCommand(0x04, data[i:i + 640])
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
        data = bytearray(20 * 16)  # 20 tiles in a row, each tile has 16 bytes representing 8x8 pixels
        if self.debug:
            print("Generating row of tiles")
        for i in range(20):  # for each tile
            for y in range(8):  # for each line within the tile
                dataoffset = 16 * i + 2 * y
                for x in range(8):  # for each column within the tile
                    pixel = pixels[160 * y + 8 * i + x]
                    if pixel & 0x40 == 0:
                        data[dataoffset] |= (0x80 >> x)
                    if pixel & 0x80 == 0:
                        data[dataoffset + 1] |= (0x80 >> x)
            if self.debug:
                print("Tile " + str(i) + ": " + " ".join(f'{c:02x}' for c in data[16 * i:16 * (i + 1)]))
        return data

    def pixelsToTiles(self, pixels):
        i = 0
        data = b''
        while i < len(pixels):
            data += self.pixelRowToTiles(pixels[i:i + 8 * 160])
            i += 8 * 160
        return data

    def printImage(self, pixels, exposure):
        print("Check printer presence and initialize.")
        self.status()
        self.initialize()
        state = self.status()
        if state != 0x00:
            raise self.PrinterException("Unexpected status after initialization: " + f'{state:02x}')

        if len(pixels) % (16 * 160) != 0:
            raise self.PrinterException("Image height must be a multiple of 16!")
        multiPartPrint = len(pixels) > 144 * 160

        if multiPartPrint:  # Manually add margin to print without margins
            print("Adding empty margin for multi-part print")
            self.initialize()
            self.fill(b'')
            self.startPrint(0, 0, 1, 0b11100100, exposure)
            self.waitForEndOfPrint()

        i = 0
        while i < len(pixels):
            print("Printing...")
            self.initialize()
            self.fill(self.pixelsToTiles(pixels[i:i + 144 * 160]))
            self.startPrint(1, 0 if multiPartPrint else 1, 0 if multiPartPrint else 3, 0b11100100, exposure)
            self.waitForEndOfPrint()
            i += 144 * 160

        if multiPartPrint:  # Manually add margin to print without margins
            print("Adding empty margin for multi-part print")
            self.initialize()
            self.fill(b'')
            self.startPrint(0, 0, 3, 0b11100100, exposure)
            self.waitForEndOfPrint()

    def printImageFromFile(self, path, exposure):
        img = Image.open(path)
        img = img.resize((160, 160 * img.size[1] // img.size[0]), Image.Resampling.LANCZOS)
        pixels = img.convert('L').tobytes()
        self.printImage(pixels, exposure)


if __name__ == "__main__":
    # The __init__ is updated, so you can call it without arguments
    # to use the default SPI bus 0, device 0.
    with GBPrinter() as printer:
        printer.printImageFromFile("captures/gameboy/gameboy_1758056969.339658.png", 0.5)