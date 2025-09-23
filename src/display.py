import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from enum import Enum


class DisplayManager:
    """Manages all display-related operations for the photobooth."""

    def __init__(self, window_name: str, fullscreen: bool, scale: float, font: ImageFont.FreeTypeFont,
                 state_enum: Enum):
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.display_scale = scale
        self.font = font
        self.PhotoboothState = state_enum  # Pass the Enum type itself

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
            display_frame = cv2.resize(frame, (0, 0), fx=self.display_scale, fy=self.display_scale)

        frame_pil = Image.fromarray(display_frame)
        draw = ImageDraw.Draw(frame_pil)

        if state == self.PhotoboothState.IDLE:
            #self._draw_text(draw, "PRESS BUTTON")
            pass

        elif state == self.PhotoboothState.COUNTDOWN:
            if countdown_remaining > 1.1:  # Buffer to show "SMILE!"
                countdown_text = str(int(np.ceil(countdown_remaining)))
                self._draw_text(draw, countdown_text)
            else:
                self._draw_text(draw, "SMILE!")

        elif state == self.PhotoboothState.CAPTURING:
            self._draw_text(draw, "CAPTURING...")

        elif state == self.PhotoboothState.PROCESSING:
            self._draw_text(draw, "WAIT...")

        display_frame = np.array(frame_pil)
        cv2.imshow(self.window_name, display_frame)

    def _draw_text(self, draw: ImageDraw.Draw, text: str):
        """Helper to draw centered text on the frame."""
        # Note: textbbox is more accurate for centering than textlength
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img_width, img_height = draw.im.size

        x = (img_width - text_width) / 2
        y = (img_height - text_height) / 4  # Position text 1/4 down the screen
        draw.text((x, y), text, font=self.font, fill=(0, 255, 0, 255))

    def destroy_windows(self):
        """Closes all OpenCV windows."""
        cv2.destroyAllWindows()
