import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from common import PhotoboothState

class DisplayManager:
    """Manages all display-related operations for the photobooth."""

    def __init__(self, window_name: str, fullscreen: bool, scale: float, header_font: ImageFont.FreeTypeFont, center_font: ImageFont.FreeTypeFont):
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.display_scale = scale
        self.header_font = header_font
        self.center_font = center_font

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def render(self, frame: np.ndarray, state: Enum, countdown_remaining: float = 0):
        """
        Renders the appropriate UI given the current application state.

        Args:
            frame: The camera preview frame.
            state: The current state of the photobooth.
            countdown_remaining: The remaining time in the countdown.
        """
        if frame is None:
            # If no frame is available, create a black screen
            width, height = (int(160 * self.display_scale), int(112 * self.display_scale))
            display_frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            width, height = 800, 600
            display_frame = np.zeros((height, width, 3), dtype=np.uint8)
            y_offset = 15
            x0 = int(width/2-64*self.display_scale)
            x1 = int(width/2+64*self.display_scale)
            y0 = int(height/2-56*self.display_scale+y_offset)
            y1 = int(height/2+56*self.display_scale+y_offset)
            frame_flip = cv2.flip(frame, 1)
            display_frame[y0:y1, x0:x1] = cv2.resize(frame_flip, (0, 0), fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_NEAREST)

        frame_pil = Image.fromarray(display_frame)
        draw = ImageDraw.Draw(frame_pil)

        if state == PhotoboothState.IDLE:
            self._draw_header_text(draw, "PRESS RED BUTTON TO START")
            pass

        elif state == PhotoboothState.COUNTDOWN:
            if countdown_remaining > 1.1:  # Buffer to show "SMILE!"
                countdown_text = str(int(np.ceil(countdown_remaining)))
                self._draw_center_text(draw, countdown_text)
                self._draw_header_text(draw, "LOOK AT THE CAMERAS!")
            else:
                self._draw_center_text(draw, "SMILE!")
                self._draw_header_text(draw, "LOOK AT THE CAMERAS!")


        #elif state == PhotoboothState.CAPTURING:
        #    self._draw_text(draw, "CAPTURING...")

        elif state == PhotoboothState.PROCESSING:
            self._draw_header_text(draw, "PROCESSING...")

        elif state == PhotoboothState.PRINTING:
            self._draw_header_text(draw, "PRINTING...")

        display_frame = np.array(frame_pil)
        cv2.imshow(self.window_name, display_frame)

    def _draw_header_text(self, draw: ImageDraw.ImageDraw, text: str):
        """Helper to draw centered text on the frame."""
        bbox = draw.textbbox((0, 0), text, font=self.header_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img_width, img_height = draw.im.size

        x = (img_width - text_width) / 2
        y = 20
        draw.text((x, y), text, font=self.header_font, fill=(0, 255, 0, 255))


    def _draw_center_text(self, draw: ImageDraw.ImageDraw, text: str):
        """Helper to draw centered text on the frame."""
        bbox = draw.textbbox((0, 0), text, font=self.center_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img_width, img_height = draw.im.size

        x = (img_width - text_width) / 2
        y = 250
        draw.text((x, y), text, font=self.center_font, fill=(0, 255, 0, 255))

    @staticmethod
    def destroy_windows():
        """Closes all OpenCV windows."""
        cv2.destroyAllWindows()
