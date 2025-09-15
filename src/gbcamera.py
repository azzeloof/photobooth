import cv2
import numpy as np
import time
import threading
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from common import PhotoboothError, PhotoboothImage, ImageMetadata, ImageManager

logger = logging.getLogger(__name__)

@dataclass
class GBCameraConfig:
    """Configuration for Game Boy camera effects"""
    # Crop settings
    crop_start_x: int = 128
    crop_start_y: int = 128
    crop_width: int = 128
    crop_height: int = 112
    crop_scale: int = 8

    # Quantization levels (Game Boy has 4 shades)
    quantization_levels: Tuple[int, int, int, int] = (0, 96, 178, 255)
    quantization_thresholds: Tuple[int, int, int] = (64, 128, 192)

    # Colors for Game Boy green palette
    gb_colors: Tuple[Tuple[int, int, int], ...] = (
        (41, 65, 57),  # Darkest green
        (57, 89, 74),  # Dark green
        (90, 121, 66),  # Light green
        (123, 130, 16)  # Lightest green
    )

    # File paths
    border_image_path: str = "gb_border.png"
    capture_directory: str = "captures"


class GBCameraError(PhotoboothError):
    """Custom exception for GB Camera errors"""
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message, recoverable, component="GBCamera")


class GBCamera:
    def __init__(self, device: int = 0, config: Optional[GBCameraConfig] = None):
        self.config = config or GBCameraConfig()
        self.device = device

        # Camera state
        self.cam: Optional[cv2.VideoCapture] = None
        self.running = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.RLock()

        # Camera properties
        self.frame_width = 0
        self.frame_height = 0

        # Border image cache
        self._border_image: Optional[np.ndarray] = None

        self.image_manager = ImageManager(self.config.capture_directory)

        self._initialize_camera()
        self._load_border_image()
        self._ensure_capture_directory()

    def _initialize_camera(self):
        """Initialize the camera with proper error handling"""
        try:
            self.cam = cv2.VideoCapture(self.device)

            if not self.cam.isOpened():
                raise GBCameraError(f"Could not open camera device {self.device}")

            # Get camera properties
            self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Test camera by reading a frame
            ret, test_frame = self.cam.read()
            if not ret or test_frame is None:
                raise GBCameraError("Camera is not providing frames")

            logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")

        except Exception as error:
            if self.cam:
                self.cam.release()
                self.cam = None
            raise GBCameraError(f"Camera initialization failed: {error}")

    def _load_border_image(self):
        """Load and validate the border image"""
        try:
            if os.path.exists(self.config.border_image_path):
                self._border_image = cv2.imread(self.config.border_image_path)
                if self._border_image is None:
                    logger.warning(f"Could not load border image: {self.config.border_image_path}")
                else:
                    logger.info(f"Border image loaded: {self._border_image.shape}")
            else:
                logger.warning(f"Border image not found: {self.config.border_image_path}")
        except Exception as error:
            logger.error(f"Error loading border image: {error}")

    def _ensure_capture_directory(self):
        """Ensure the capture directory exists"""
        try:
            Path(self.config.capture_directory).mkdir(parents=True, exist_ok=True)
        except Exception as error:
            logger.error(f"Could not create capture directory: {error}")

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop frame to Game Boy camera dimensions with validation

        Args:
            frame: Input frame to crop

        Returns:
            Cropped and resized frame

        Raises:
            GBCameraError: If the frame is invalid or cropping fails
        """
        if frame is None:
            raise GBCameraError("Cannot crop None frame")

        try:
            height, width = frame.shape[:2]

            # Calculate crop boundaries
            end_x = self.config.crop_start_x + (self.config.crop_width * self.config.crop_scale)
            end_y = self.config.crop_start_y + (self.config.crop_height * self.config.crop_scale)

            # Validate crop boundaries
            if (end_x > width or end_y > height or
                    self.config.crop_start_x < 0 or self.config.crop_start_y < 0):
                logger.warning(f"Crop boundaries exceed frame size {width}x{height}, adjusting...")

                # Adjust to fit within the frame
                self.config.crop_start_x = max(0, min(self.config.crop_start_x, width - 100))
                self.config.crop_start_y = max(0, min(self.config.crop_start_y, height - 100))
                end_x = min(end_x, width)
                end_y = min(end_y, height)

            # Perform crop
            cropped = frame[self.config.crop_start_y:end_y,
            self.config.crop_start_x:end_x]

            # Resize to Game Boy dimensions
            resized = cv2.resize(cropped,
                                 (self.config.crop_width, self.config.crop_height),
                                 interpolation=cv2.INTER_AREA)

            return resized

        except Exception as error:
            raise GBCameraError(f"Cropping failed: {error}")

    def quantize(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Game Boy 4-level quantization

        Args:
            frame: Input frame to quantize

        Returns:
            Quantized frame with 4 gray levels

        Raises:
            GBCameraError: If quantization fails
        """
        if frame is None:
            raise GBCameraError("Cannot quantize None frame")

        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply quantization using vectorized operations
            quantized = np.zeros_like(frame)

            # Use config values for thresholds and levels
            t1, t2, t3 = self.config.quantization_thresholds
            l0, l1, l2, l3 = self.config.quantization_levels

            quantized = np.select([
                frame <= t1,
                frame <= t2,
                frame <= t3,
                frame > t3
            ], [l0, l1, l2, l3], default=l3)

            # Convert back to 3-channel for consistency
            if len(frame.shape) == 2:
                quantized = cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            return quantized

        except Exception as error:
            raise GBCameraError(f"Quantization failed: {error}")

    def colorize(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply Game Boy green color palette

        Args:
            frame: Quantized grayscale frame

        Returns:
            Colorized frame or None if input is None
        """
        if frame is None:
            return None

        try:
            if len(frame.shape) != 3:
                # Convert to 3-channel if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            height, width, channels = frame.shape
            colored_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Map each quantization level to its corresponding color
            for i, (gray_val, color) in enumerate(zip(self.config.quantization_levels,
                                                      self.config.gb_colors)):
                # Create the mask for this gray level
                mask = np.all(frame == [gray_val, gray_val, gray_val], axis=-1)
                colored_frame[mask] = color

            return colored_frame

        except Exception as error:
            logger.error(f"Colorization failed: {error}")
            return frame  # Return original on error

    def add_border(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a Game Boy border to the frame

        Args:
            frame: Frame to add border to (must be 128x112)

        Returns:
            Frame with a border added

        Raises:
            GBCameraError: If frame dimensions are wrong or border unavailable
        """
        if frame is None:
            raise GBCameraError("Cannot add border to None frame")

        # Validate frame dimensions
        expected_height, expected_width = self.config.crop_height, self.config.crop_width
        if frame.shape[:2] != (expected_height, expected_width):
            raise GBCameraError(
                f"Frame dimensions {frame.shape[:2]} don't match expected "
                f"{(expected_height, expected_width)}"
            )

        if self._border_image is None:
            logger.warning("No border image available, returning frame as-is")
            return frame

        try:
            # Create a copy of the border to avoid modifying the cached version
            bordered_frame = self._border_image.copy()

            # Insert the frame into the border (assuming 16 px offset)
            border_offset_x, border_offset_y = 16, 16
            bordered_frame[border_offset_y:border_offset_y + frame.shape[0],
            border_offset_x:border_offset_x + frame.shape[1]] = frame

            return bordered_frame

        except Exception as error:
            raise GBCameraError(f"Border addition failed: {error}")

    def run(self):
        """
        Main camera loop - continuously captures and processes frames
        """
        if not self.cam:
            raise GBCameraError("Camera not initialized")

        self.running = True
        logger.info("Starting camera capture loop")

        try:
            while self.running:
                ret, raw_frame = self.cam.read()

                if not ret or raw_frame is None:
                    logger.error("Failed to read frame from camera")
                    time.sleep(0.1)  # Brief pause before retry
                    continue

                try:
                    # Process frame
                    processed_frame = self.crop(raw_frame)
                    processed_frame = self.quantize(processed_frame)

                    # Thread-safe frame update
                    with self.frame_lock:
                        self.current_frame = processed_frame

                except GBCameraError as error:
                    logger.error(f"Frame processing error: {error}")
                    # Continue with the next frame
                    continue
                except Exception as error:
                    logger.error(f"Unexpected error in frame processing: {error}")
                    continue

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

        except Exception as error:
            logger.error(f"Camera loop error: {error}")
        finally:
            logger.info("Camera capture loop stopped")

    def get_frame(self, color: bool = False) -> Optional[np.ndarray]:
        """
        Get the current processed frame

        Args:
            color: If True, return the colorized version

        Returns:
            Current frame or None if unavailable
        """
        with self.frame_lock:
            if self.current_frame is None:
                return None

            frame = self.current_frame.copy()  # Return copy to avoid threading issues

        if color:
            return self.colorize(frame)
        return frame

    def get_current_image(self, color: bool = False, apply_border: bool = False) -> Optional[PhotoboothImage]:
        """
        Get current frame as PhotoboothImage object

        Args:
            color: Apply Game Boy colorization
            apply_border: Add a Game Boy border

        Returns:
            PhotoboothImage or None if unavailable
        """
        with self.frame_lock:
            if self.current_frame is None:
                return None

            frame_data = self.current_frame.copy()

        # Create metadata
        metadata = ImageMetadata(
            camera_type="gameboy",
            processing_applied=[]
        )

        # Create base image
        image = PhotoboothImage.from_array(frame_data, metadata)

        # Apply colorization if requested
        if color:
            image = image.apply_processing(
                lambda data: self.colorize(data) or data,
                "gameboy_colorize"
            )

        # Apply border if requested
        if apply_border:
            image = image.apply_processing(
                self.add_border,
                "gameboy_border"
            )

        return image

    def capture_image(self, session_id: Optional[str] = None) -> PhotoboothImage:
        """
        Capture the current frame as PhotoboothImage

        Args:
            session_id: Optional session identifier

        Returns:
            PhotoboothImage with captured data

        Raises:
            GBCameraError: If capture fails
        """
        image = self.get_current_image(color=False, apply_border=False)
        if image is None:
            raise GBCameraError("No frame available for capture")

        # Update metadata
        image.metadata.session_id = session_id
        image.metadata.timestamp = time.time()

        return image

    def capture(self, session_id: Optional[str] = None) -> Tuple[PhotoboothImage, PhotoboothImage]:
        """
        Capture and return both regular and bordered versions

        Returns:
            Tuple of (regular_image, bordered_image)
        """
        base_image = self.capture_image(session_id)

        # Create bordered version
        try:
            bordered_image = base_image.apply_processing(
                self.add_border,
                "gameboy_border"
            )
        except GBCameraError as e:
            logger.warning(f"Could not create bordered version: {e}")
            bordered_image = base_image.copy()

        return (base_image, bordered_image)

    # Backwards compatibility method
    def capture_to_files(self, session_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Capture and save to files (backwards compatibility)

        Returns:
            Tuple of file paths (regular, bordered)
        """
        regular_image, bordered_image = self.capture(session_id)

        # Save both images
        timestamp = int(time.time() * 1000)
        regular_path = regular_image.save(
            os.path.join(self.config.capture_directory, f'frame_{timestamp}.png')
        )
        bordered_path = bordered_image.save(
            os.path.join(self.config.capture_directory, f'frameBordered_{timestamp}.png')
        )

        return (regular_path, bordered_path)


    def stop(self):
        """
        Stop the camera and cleanup resources
        """
        logger.info("Stopping camera...")
        self.running = False

        if self.cam:
            self.cam.release()
            self.cam = None

        with self.frame_lock:
            self.current_frame = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

    @property
    def is_initialized(self) -> bool:
        """Check if the camera is properly initialized"""
        return self.cam is not None and self.cam.isOpened()

    @property
    def camera_resolution(self) -> Tuple[int, int]:
        """Get camera resolution"""
        return self.frame_width, self.frame_height


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        with GBCamera() as camera:
            # Start the camera thread
            camera_thread = threading.Thread(target=camera.run, daemon=True)
            camera_thread.start()

            # Simple test loop
            print("Press 'c' to capture, 'q' to quit")
            cv2.namedWindow("GB Camera", cv2.WINDOW_AUTOSIZE)

            while True:
                current_frame = camera.get_frame(color=True)
                if current_frame is not None:
                    cv2.imshow("GB Camera", cv2.resize(current_frame, (0, 0), fx=4, fy=4))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    try:
                        files = camera.capture()
                        print(f"Captured: {files}")
                    except GBCameraError as e:
                        print(f"Capture failed: {e}")

            cv2.destroyAllWindows()

    except GBCameraError as error:
        print(f"Camera error: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")