import concurrent.futures
import cv2
import logging
import os
import queue
import threading
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from typing import Optional, List, Tuple
from common import SystemStatus, PhotoboothImage, ImageManager, ImageMetadata, PhotoboothState
from ds40 import DS40
from gbcamera import GBCamera, GBCameraConfig, GBCameraError
from gbprinter import GBPrinter
from hardware import Hardware, HardwareConfig, HardwareError
from nikon import NikonCamera, NikonConfig, NikonError
from display import DisplayManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

overlay_font = ImageFont.truetype("fonts/Jersey_10/Jersey10-Regular.ttf", 128)

@dataclass
class PhotoboothConfig:
    """Central configuration for the entire photobooth system"""
    # Session settings
    photos_per_session: int = 3
    delay_between_photos: float = 5.0
    countdown_duration: float = 3.0
    window_name: str = 'Photobooth'

    # Display settings
    fullscreen: bool = True
    display_scale: float = 4.0

    # Camera settings
    gb_camera_config: Optional[GBCameraConfig] = None
    nikon_config: Optional[NikonConfig] = None
    hardware_config: Optional[HardwareConfig] = None

    def __post_init__(self):
        # Set default configurations if not provided
        if self.gb_camera_config is None:
            self.gb_camera_config = GBCameraConfig()
        if self.nikon_config is None:
            self.nikon_config = NikonConfig()
        if self.hardware_config is None:
            self.hardware_config = HardwareConfig()


@dataclass
class StartSessionEvent:
    """Event to start a new photo session."""
    pass


@dataclass
class CapturePhotoEvent:
    """Event to trigger a single photo capture."""
    pass


@dataclass
class CaptureCompleteEvent:
    """Event fired when a photo capture is successfully completed."""
    result: Tuple[PhotoboothImage, PhotoboothImage, PhotoboothImage]


@dataclass
class NextPhotoEvent:
    """Event to start the countdown for the next photo in a session."""
    pass


@dataclass
class PrintEvent:
    """Event for print operations"""
    photos: List[Tuple[PhotoboothImage, PhotoboothImage, PhotoboothImage]]


@dataclass
class ErrorEvent:
    """Event for error conditions"""
    message: str
    recoverable: bool = True


class PhotoSet:
    def __init__(self, max_photos: int, delay_between_photos: float = 5.0):
        self.created_at = time.time()
        self.max_photos = max_photos
        self.delay_between_photos = delay_between_photos
        self.captures: List[
            Tuple[PhotoboothImage, PhotoboothImage, PhotoboothImage]] = []  # (gb_image, gb_ai_image, nikon_image) pairs
        self.current_capture = 0
        self.session_id = f"session_{int(time.time())}"

    def add_capture(self, gb_image: PhotoboothImage, gb_ai_image: PhotoboothImage,
                    nikon_image: PhotoboothImage) -> bool:
        """Add a capture to the set. Returns True if successful, False if the set is full."""
        if len(self.captures) < self.max_photos:
            self.captures.append((gb_image, gb_ai_image, nikon_image))
            self.current_capture += 1
            return True
        return False

    def is_complete(self) -> bool:
        return len(self.captures) >= self.max_photos

    def remaining_photos(self) -> int:
        return self.max_photos - len(self.captures)

    def save_metadata(self) -> None:
        """Save metadata about this photo set"""
        # Implementation for saving session info
        pass


class CameraManager:
    """Manages camera operations in a thread-safe manner"""

    def __init__(self, local_config: PhotoboothConfig):
        self.config = local_config
        self.image_manager = ImageManager("captures")
        self.gb_camera: Optional[GBCamera] = None
        self.nikon: Optional[NikonCamera] = None
        self.gb_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize cameras with proper error handling"""
        try:
            # Initialize GB Camera
            self.gb_camera = GBCamera(config=self.config.gb_camera_config)
            logger.info(f"GB Camera initialized: {self.gb_camera.camera_resolution}")

            # Initialize Nikon Camera
            self.nikon = NikonCamera(config=self.config.nikon_config)
            logger.info(f"Nikon camera initialized: {self.nikon.camera_model}")

            # Start GB camera thread
            self.gb_thread = threading.Thread(target=self.gb_camera.run, daemon=True)
            self.gb_thread.start()

            self._initialized = True
            logger.info("Cameras initialized successfully")
            return True

        except (GBCameraError, NikonError) as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during camera initialization: {e}")
            return False

    def capture_photos(self, session_id: str) -> Optional[Tuple[PhotoboothImage, PhotoboothImage, PhotoboothImage]]:
        """Thread-safe photo capture returning PhotoboothImage objects"""
        if not self._initialized:
            return None

        with self._lock:
            try:
                # Capture from both cameras as PhotoboothImages
                gb_regular, gb_bordered = self.gb_camera.capture(session_id)
                nikon_image = self.nikon.capture_image(session_id)

                # Crop Nikon image to match GB aspect ratio (160/144 = 1.111...)
                nikon_cropped = self._crop_to_gb_aspect_ratio(nikon_image)
                return gb_regular, gb_bordered, nikon_cropped

            except (GBCameraError, NikonError) as e:
                logger.error(f"Capture failed: {e}")
                return None

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get the current preview frame as a numpy array for display"""
        if self.gb_camera and self.gb_camera.is_initialized:
            preview_image = self.gb_camera.get_current_image(color=True)
            return preview_image.data if preview_image else None
        return None

    @staticmethod
    def _crop_to_gb_aspect_ratio(image: PhotoboothImage) -> PhotoboothImage:
        """
        Crop image to match the Game Boy camera aspect ratio (160/144)

        Args:
            image: PhotoboothImage to crop

        Returns:
            Cropped PhotoboothImage with the Game Boy aspect ratio
        """
        target_aspect_ratio = 160.0 / 144.0  # GB camera aspect ratio

        # Get current image dimensions
        height, width = image.shape[:2]
        current_aspect_ratio = width / height

        if abs(current_aspect_ratio - target_aspect_ratio) < 0.001:
            # Already the correct aspect ratio
            return image

        if current_aspect_ratio > target_aspect_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_aspect_ratio)
            x_offset = (width - new_width) // 2
            cropped = image.crop(x_offset, 0, new_width, height)
        else:
            # Image is too tall, crop height
            new_height = int(width / target_aspect_ratio)
            y_offset = (height - new_height) // 2
            cropped = image.crop(0, y_offset, width, new_height)

        logger.info(
            f"Cropped Nikon image from {width}x{height} to {cropped.shape[1]}x{cropped.shape[0]} for GB aspect ratio")
        return cropped

    def shutdown(self):
        """Cleanup cameras"""
        logger.info("Shutting down cameras...")

        if self.gb_camera:
            self.gb_camera.stop_thread()

        if self.gb_thread and self.gb_thread.is_alive():
            self.gb_thread.join(timeout=2.0)
            logger.info("GB camera thread joined.")

        if self.gb_camera:
            self.gb_camera.release()

        if self.nikon:
            self.nikon.release()

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class Photobooth:
    def __init__(self, local_config: Optional[PhotoboothConfig] = None):
        self.config = local_config or PhotoboothConfig()
        self.system_status = SystemStatus()

        # State management
        self.state = PhotoboothState.IDLE
        self.event_queue = queue.Queue()
        self.current_photo_set: Optional[PhotoSet] = None

        # Hardware components
        self.camera_manager = CameraManager(self.config)
        self.printer = DS40()
        self.gb_printer = GBPrinter()
        self.hardware: Optional[Hardware] = None
        self.display_manager = DisplayManager(
            self.config.window_name,
            self.config.fullscreen,
            self.config.display_scale,
            overlay_font
        )

        # Threading
        self.running = False
        self.state_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Timing
        self.countdown_end_time = 0

        # Initialize components
        self._initialize_hardware()
        self._initialize_cameras()

    def _initialize_hardware(self):
        """Initialize hardware components"""
        try:
            self.hardware = Hardware(self.config.hardware_config)
            self.hardware.register_callback(0, self._on_capture_button)
            self.system_status.hardware_ready = True
            logger.info("Hardware initialized")
        except HardwareError as e:
            self.system_status.add_error(f"Hardware error: {e}")
            logger.warning(f"Hardware not available: {e}")
            self.hardware = None
        except Exception as e:
            self.system_status.add_error(f"Unexpected hardware error: {e}")
            logger.warning(f"Unexpected hardware error: {e}")
            self.hardware = None

    def _initialize_cameras(self):
        """Initialize camera system"""
        if self.camera_manager.initialize():
            self.system_status.cameras_ready = True
        else:
            self.system_status.add_error("Camera initialization failed")
            self._transition_to_error("Camera initialization failed")

    def _on_capture_button(self):
        """Hardware button callback"""
        self.event_queue.put(StartSessionEvent())

    def _transition_to_error(self, message: str):
        """Transition to error state"""
        with self.state_lock:
            self.state = PhotoboothState.ERROR
            logger.error(f"Error state: {message}")

    def run(self):
        """Main application loop"""
        self.running = True

        if self.hardware:
            self.hardware.start()

        try:
            while self.running:
                self._process_events()
                self._render_display()
                self._handle_keyboard_input()
                time.sleep(0.016)  # ~60 FPS

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._shutdown()

    def _process_events(self):
        """Process events from the queue"""
        try:
            event = self.event_queue.get_nowait()
            self._handle_event(event)
        except queue.Empty:
            pass

    def _handle_event(self, event):
        """Central hub for all state transitions and logic"""
        with self.state_lock:
            if isinstance(event, StartSessionEvent) and self.state == PhotoboothState.IDLE:
                self._start_capture_session()

            elif isinstance(event, CapturePhotoEvent) and self.state == PhotoboothState.COUNTDOWN:
                self._capture_photo()

            elif isinstance(event, CaptureCompleteEvent):
                self._handle_capture_result(event.result)

            elif isinstance(event, NextPhotoEvent) and self.state == PhotoboothState.PROCESSING:
                self.state = PhotoboothState.COUNTDOWN
                self._start_countdown_timer()

            elif isinstance(event, PrintEvent):
                self._handle_print_event(event)

            elif isinstance(event, ErrorEvent):
                self._transition_to_error(event.message)

    def _start_capture_session(self):
        """Start a new multi-photo capture session"""
        self.current_photo_set = PhotoSet(
            self.config.photos_per_session,
            self.config.delay_between_photos
        )
        self.state = PhotoboothState.COUNTDOWN
        logger.info(f"Starting capture session: {self.config.photos_per_session} photos")

        if self.hardware:
            self.hardware.blink_button_led(0, 0.5)

        self._start_countdown_timer()

    def _start_countdown_timer(self):
        """Starts a non-blocking timer to trigger the capture."""
        self.countdown_end_time = time.time() + self.config.countdown_duration
        # Use a timer to put the capture event on the queue, making it non-blocking
        threading.Timer(self.config.countdown_duration,
                        lambda: self.event_queue.put(CapturePhotoEvent())).start()

    def _capture_photo(self):
        """Initiate photo capture and transition to the PROCESSING state."""
        self.state = PhotoboothState.CAPTURING
        session_id = self.current_photo_set.session_id if self.current_photo_set else None

        # Submit the capture task and add a callback to handle the result
        future = self.executor.submit(self.camera_manager.capture_photos, session_id)
        future.add_done_callback(self._on_capture_future_done)
        self.state = PhotoboothState.PROCESSING  # Show "PLEASE WAIT..." immediately

    def _on_capture_future_done(self, future):
        """Callback executed when the capture task finishes. Puts result in the event queue."""
        try:
            result = future.result()
            if result:
                self.event_queue.put(CaptureCompleteEvent(result=result))
            else:
                self.event_queue.put(ErrorEvent("Photo capture failed"))
        except Exception as e:
            self.event_queue.put(ErrorEvent(f"Capture error: {e}"))

    def _handle_capture_result(self, result):
        """Handle a completed photo capture"""
        if self.current_photo_set:
            gb_regular, gb_bordered, nikon_cropped = result
            _, gb_ai_bordered = self._create_ai_upscaled_version(gb_regular)

            self.current_photo_set.add_capture(gb_bordered, gb_ai_bordered, nikon_cropped)
            logger.info(
                f"Captured photo {self.current_photo_set.current_capture}/{self.config.photos_per_session}")

            if self.hardware:
                self.hardware.blink_button_led(0, 0.2)

            if self.current_photo_set.is_complete():
                self._start_printing()
            else:
                # Start a timer for the delay between photos
                threading.Timer(self.config.delay_between_photos,
                                lambda: self.event_queue.put(NextPhotoEvent())).start()

    def _start_printing(self):
        """Start the printing process"""
        self.state = PhotoboothState.PRINTING
        if self.current_photo_set:
            self.event_queue.put(PrintEvent(self.current_photo_set.captures))
            self.current_photo_set = None
            self.state = PhotoboothState.IDLE

    @staticmethod
    def _create_ai_upscaled_version(photo: PhotoboothImage):
        try:
            border_image = cv2.imread("gb_ai_border.png")
        except Exception as e:
            logger.error(f"AI border image load error: {e}")
            return None, None

        ai_upscaled = cv2.resize(photo.data.copy(), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        border_image[96:96 + ai_upscaled.shape[0], 96:96 + ai_upscaled.shape[1]] = ai_upscaled

        # Create PhotoboothImage objects with copied metadata to preserve timestamp
        ai_upscaled_metadata = ImageMetadata(
            timestamp=photo.metadata.timestamp,
            camera_type="gameboy_ai",
            session_id=photo.metadata.session_id,
            image_index=photo.metadata.image_index,
            processing_applied=photo.metadata.processing_applied.copy() + ["ai_upscale"]
        )

        ai_bordered_metadata = ImageMetadata(
            timestamp=photo.metadata.timestamp,
            camera_type="gameboy_ai_framed",
            session_id=photo.metadata.session_id,
            image_index=photo.metadata.image_index,
            processing_applied=photo.metadata.processing_applied.copy() + ["ai_upscale", "border"]
        )

        ai_upscaled_image = PhotoboothImage(data=ai_upscaled, metadata=ai_upscaled_metadata)
        ai_upscaled_image_bordered = PhotoboothImage(data=border_image, metadata=ai_bordered_metadata)

        # Save AI upscaled images with proper paths
        timestamp_ms = int(photo.metadata.timestamp * 1000)
        ai_filepath = os.path.join("captures", "gameboy", f"gameboy_ai_{timestamp_ms}.png")
        ai_framed_filepath = os.path.join("captures", "gameboy", f"gameboy_ai_framed_{timestamp_ms}.png")

        ai_upscaled_image.save(ai_filepath)
        ai_upscaled_image_bordered.save(ai_framed_filepath)

        return ai_upscaled_image, ai_upscaled_image_bordered

    def _handle_print_event(self, event: PrintEvent):
        """Handle print events"""
        try:
            filename = os.path.join("captures", "composite", f"session_{int(time.time())}.pdf")
            pages = layout_page(event.photos)
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as multi-page PDF
            pages[0].save(filename, "PDF", resolution=100.0, save_all=True, append_images=pages[1:])
            self.printer.print(filename)
        except Exception as e:
            logger.error(f"Printing failed: {e}")

    def _render_display(self):
        """Delegates all rendering tasks to the DisplayManager."""
        frame = self.camera_manager.get_preview_frame()

        # Calculate the remaining time for the countdown if applicable
        countdown_remaining = 0.0
        if self.state == PhotoboothState.COUNTDOWN:
            countdown_remaining = max(0.0, self.countdown_end_time - time.time())

        self.display_manager.render(frame, self.state, countdown_remaining)

    def _handle_keyboard_input(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('c'):
            self.event_queue.put(StartSessionEvent())
        elif key == ord('r') and self.state == PhotoboothState.ERROR:
            # Reset from an error state
            with self.state_lock:
                self.state = PhotoboothState.IDLE

    def _shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down photobooth...")

        if self.hardware:
            self.hardware.stop()

        self.camera_manager.shutdown()
        self.display_manager.destroy_windows()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.warning(f"Executor shutdown warning: {e}")


def layout_page(frames: List[Tuple[PhotoboothImage, PhotoboothImage, PhotoboothImage]]):
    """
    Lay the full page out to be printed. Creates two 4"x6" pages with two columns per page,
    handling four photo sessions total.
    :param frames: List[Tuple[gb_image: PhotoboothImage, gb_ai_image: PhotoboothImage, nikon_image: PhotoboothImage]]
                   Expected to have 4 elements
    :return: List of PIL Images, one for each page
    """
    ######### Parameters #########
    page_width = 1200  # 4 inches * 300 DPI
    page_height = 1800  # 6 inches * 300 DPI
    
    # Simple margin control
    margin_x = 50  # Side margins
    margin_y_top = 50  # Top margin
    margin_y_bottom = 350  # Bottom margin (for text space)
    
    # Image aspect ratio (GB camera: 160/144)
    image_aspect_ratio = 160.0 / 144.0  # 1.111...
    
    # Calculate available space
    available_width = page_width - (2 * margin_x)
    available_height = page_height - margin_y_top - margin_y_bottom
    
    # Column layout
    column_spacing = 50  # Space between columns
    max_column_width = (available_width - column_spacing) // 2
    
    # Calculate image dimensions based on both constraints
    min_spacing_between_images = 20  # Minimum spacing between images
    max_total_image_height = available_height - (2 * min_spacing_between_images)
    max_image_height_from_y = max_total_image_height // 3
    
    # Find the most restrictive constraint
    max_image_width_from_x = max_column_width
    max_image_height_from_x = int(max_image_width_from_x / image_aspect_ratio)
    
    # Use the most restrictive dimension
    if max_image_height_from_x <= max_image_height_from_y:
        # X dimension is more restrictive
        image_width = max_image_width_from_x
        image_height = max_image_height_from_x
    else:
        # Y dimension is more restrictive
        image_height = max_image_height_from_y
        image_width = int(image_height * image_aspect_ratio)
    
    # Calculate actual spacing between images to center them vertically
    total_images_height = 3 * image_height
    remaining_height = available_height - total_images_height
    spacing_between_images = remaining_height / 2  # 2 gaps between 3 images
    
    ##############################
    assert (len(frames) == 4)
    assert (len(frames[0]) == 3)

    logger.info(f"Laying out two pages with {len(frames)} frames")
    logger.info(f"Image size: {image_width}x{image_height}, Spacing: {spacing_between_images}")

    # Create two separate pages
    page1 = Image.new('RGB', (page_width, page_height), 'white')
    page2 = Image.new('RGB', (page_width, page_height), 'white')
    pages = [page1, page2]

    # Calculate column starting positions (centered in their available space)
    column_area_width = (available_width - column_spacing) // 2
    col_x = [
        margin_x + (column_area_width - image_width) // 2,  # Center first column
        margin_x + column_area_width + column_spacing + (column_area_width - image_width) // 2  # Center second column
    ]

    # Process frames in pairs: first two frames go on page 1, and the next two on page 2
    for page_idx, page in enumerate(pages):
        frame_start = page_idx * 2
        frame_end = frame_start + 2
        page_frames = frames[frame_start:frame_end]
        
        for col_idx, (gb_image, gb_ai_image, nikon_image) in enumerate(page_frames):
            images_by_type = [gb_image, gb_ai_image, nikon_image]

            for image_type, image in enumerate(images_by_type):
                # Calculate Y position with automatic spacing
                x = col_x[col_idx]
                y = margin_y_top + image_type * (image_height + spacing_between_images)

                try:
                    img = Image.open(image.file_path)
                    # Resize to calculated dimensions
                    img = img.resize((image_width, image_height))
                    
                    # Paste image at calculated position
                    page.paste(img, (x, int(y)))
                except Exception as e:
                    logger.error(f"Error placing image {image}: {e}")
                    
    return pages


def test_layout():
    cm = CameraManager(PhotoboothConfig())
    frames = [
        (
            PhotoboothImage.from_file("captures/gameboy/gb1.png"),
            PhotoboothImage.from_file("captures/gameboy/gb_ai1.png"),
            cm._crop_to_gb_aspect_ratio(PhotoboothImage.from_file("captures/nikon/nikon1.jpg"))
        ),
        (
            PhotoboothImage.from_file("captures/gameboy/gb2.png"),
            PhotoboothImage.from_file("captures/gameboy/gb_ai2.png"),
            cm._crop_to_gb_aspect_ratio(PhotoboothImage.from_file("captures/nikon/nikon2.jpg"))
        ),
        (
            PhotoboothImage.from_file("captures/gameboy/gb3.png"),
            PhotoboothImage.from_file("captures/gameboy/gb_ai3.png"),
            cm._crop_to_gb_aspect_ratio(PhotoboothImage.from_file("captures/nikon/nikon3.jpg"))
        ),
        (
            PhotoboothImage.from_file("captures/gameboy/gb1.png"),
            PhotoboothImage.from_file("captures/gameboy/gb_ai1.png"),
            cm._crop_to_gb_aspect_ratio(PhotoboothImage.from_file("captures/nikon/nikon1.jpg"))
        )
    ]
    pages = layout_page(frames)
    pages[0].save("captures/test.pdf", "PDF", resolution=100.0, save_all=True, append_images=pages[1:])


if __name__ == "__main__":
    # Example with custom configuration
    config = PhotoboothConfig(
        photos_per_session=4,
        delay_between_photos=5.0,
        countdown_duration=5.0,
        gb_camera_config=GBCameraConfig(
            crop_start_x=511,
            crop_start_y=147
        ),
        nikon_config=NikonConfig(
            iso=800,
            image_format="JPEG"
        )
    )

    #photobooth = Photobooth(config)
    #photobooth.run()
    test_layout()