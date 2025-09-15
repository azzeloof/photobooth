import concurrent.futures
import cv2
import logging
import queue
import threading
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple
from common import SystemStatus, PhotoboothImage, ImageManager
from ds40 import DS40
from gbcamera import GBCamera, GBCameraConfig, GBCameraError
from gbprinter import GBPrinter
from hardware import Hardware, HardwareConfig, HardwareError
from nikon import NikonCamera, NikonConfig, NikonError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotoboothState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    CAPTURING = auto()
    PROCESSING = auto()
    PRINTING = auto()
    ERROR = auto()


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
class CaptureEvent:
    """Event for capture operations"""
    pass


@dataclass
class PrintEvent:
    """Event for print operations"""
    photos: List[Tuple[str, str]]


@dataclass
class CountdownEvent:
    """Event for countdown operations"""
    remaining_time: float


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
        self.captures: List[Tuple[PhotoboothImage, PhotoboothImage]] = []  # (gb_image, nikon_image) pairs
        self.current_capture = 0
        self.session_id = f"session_{int(time.time())}"

    def add_capture(self, gb_image: PhotoboothImage, nikon_image: PhotoboothImage) -> bool:
        """Add a capture to the set. Returns True if successful, False if set is full."""
        if len(self.captures) < self.max_photos:
            self.captures.append((gb_image, nikon_image))
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

    def capture_photos(self, session_id: str) -> Optional[Tuple[PhotoboothImage, PhotoboothImage]]:
        """Thread-safe photo capture returning PhotoboothImage objects"""
        if not self._initialized:
            return None

        with self._lock:
            try:
                # Capture from both cameras as PhotoboothImages
                gb_regular, gb_bordered = self.gb_camera.capture(session_id)
                nikon_image = self.nikon.capture_image(session_id)
                return (gb_regular, nikon_image)

            except (GBCameraError, NikonError) as e:
                logger.error(f"Capture failed: {e}")
                return None

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get current preview frame as numpy array for display"""
        if self.gb_camera and self.gb_camera.is_initialized:
            preview_image = self.gb_camera.get_current_image(color=True)
            return preview_image.data if preview_image else None
        return None

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

        # Threading
        self.running = False
        self.state_lock = threading.Lock()

        # Timing
        self.countdown_start = 0
        self.next_capture_time = 0

        # Initialize components
        self._initialize_hardware()
        self._initialize_cameras()
        self._setup_display()

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

    def _setup_display(self):
        """Setup OpenCV display"""
        cv2.namedWindow(self.config.window_name, cv2.WND_PROP_FULLSCREEN)
        if self.config.fullscreen:
            cv2.setWindowProperty(self.config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _on_capture_button(self):
        """Hardware button callback"""
        self.event_queue.put(CaptureEvent())

    def _transition_to_error(self, message: str):
        """Transition to error state"""
        with self.state_lock:
            self.state = PhotoboothState.ERROR
            logger.error(f"Error state: {message}")

    def run(self):
        """Main application loop"""
        self.running = True

        # Start the hardware polling thread if available
        if self.hardware:
            self.hardware.start()

        try:
            while self.running:
                self._process_events()
                self._update_state()
                self._render_display()
                self._handle_keyboard_input()

                # Small delay to prevent busy waiting
                time.sleep(0.016)  # ~60 FPS

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._shutdown()

    def _process_events(self):
        """Process events from the queue"""
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass

    def _handle_event(self, event):
        """Handle individual events based on the current state"""
        with self.state_lock:
            if isinstance(event, CaptureEvent):
                if self.state == PhotoboothState.IDLE:
                    self._start_capture_session()

            elif isinstance(event, ErrorEvent):
                self._transition_to_error(event.message)

            elif isinstance(event, PrintEvent):
                self._handle_print_event(event)

    @staticmethod
    def _handle_print_event(event: PrintEvent):
        """Handle print events"""
        try:
            # TODO: Implement actual printing
            page = layout_page(event.photos)
            # self.printer.print(page)
            logger.info(f"Would print {len(event.photos)} photos")
        except Exception as e:
            logger.error(f"Printing failed: {e}")

    def _start_capture_session(self):
        """Start a new multi-photo capture session"""
        self.current_photo_set = PhotoSet(
            self.config.photos_per_session,
            self.config.delay_between_photos
        )
        self.countdown_start = time.time()
        self.state = PhotoboothState.COUNTDOWN
        logger.info(f"Starting capture session: {self.config.photos_per_session} photos")

        # Visual feedback
        if self.hardware:
            self.hardware.blink_button_led(0, 0.5)

    def _update_state(self):
        """Update state machine"""
        current_time = time.time()

        with self.state_lock:
            if self.state == PhotoboothState.COUNTDOWN:
                elapsed = current_time - self.countdown_start
                if elapsed >= self.config.countdown_duration:
                    self._capture_photo()

            elif self.state == PhotoboothState.CAPTURING:
                # Handle an ongoing capture process
                if hasattr(self, '_capture_future') and self._capture_future.done():
                    self._handle_capture_result()

    def _capture_photo(self):
        """Initiate photo capture"""
        self.state = PhotoboothState.CAPTURING

        # Use a thread pool for non-blocking capture
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self._capture_future = executor.submit(self.camera_manager.capture_photos)

    def _handle_capture_result(self):
        """Handle a completed photo capture"""
        try:
            result = self._capture_future.result()
            if result and self.current_photo_set:
                gb_file, nikon_file = result
                self.current_photo_set.add_capture(gb_file, nikon_file)
                logger.info(
                    f"Captured photo {self.current_photo_set.current_capture}/{self.config.photos_per_session}")

                # Visual feedback
                if self.hardware:
                    self.hardware.blink_button_led(0, 0.2)

                if self.current_photo_set.is_complete():
                    self._start_printing()
                else:
                    self._prepare_next_capture()
            else:
                self.event_queue.put(ErrorEvent("Photo capture failed"))
        except Exception as e:
            self.event_queue.put(ErrorEvent(f"Capture error: {e}"))

    def _prepare_next_capture(self):
        """Prepare for the next photo in the session"""
        self.next_capture_time = time.time() + self.config.delay_between_photos
        self.countdown_start = self.next_capture_time - self.config.countdown_duration
        self.state = PhotoboothState.COUNTDOWN

    def _start_printing(self):
        """Start the printing process"""
        self.state = PhotoboothState.PRINTING
        if self.current_photo_set:
            self.event_queue.put(PrintEvent(self.current_photo_set.captures))
            # Reset for the next session
            self.current_photo_set = None
            self.state = PhotoboothState.IDLE

    def _render_display(self):
        """Render the current display"""
        frame = self.camera_manager.get_preview_frame()
        if frame is not None:
            display_frame = cv2.resize(frame, (0, 0),
                                       fx=self.config.display_scale,
                                       fy=self.config.display_scale)

            # Add state-specific overlays
            self._add_state_overlay(display_frame)

            cv2.imshow(self.config.window_name, display_frame)

    def _add_state_overlay(self, frame):
        """Add overlay information based on the current state"""
        if self.state == PhotoboothState.COUNTDOWN:
            remaining = self.config.countdown_duration - (time.time() - self.countdown_start)
            if remaining > 0:
                countdown_text = str(max(1, int(remaining)))
                cv2.putText(frame, countdown_text, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        elif self.state == PhotoboothState.CAPTURING:
            cv2.putText(frame, "SMILE!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        elif self.state == PhotoboothState.ERROR:
            cv2.putText(frame, "ERROR - Press C to retry", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif self.current_photo_set:
            progress_text = f"Photo {self.current_photo_set.current_capture}/{self.config.photos_per_session}"
            cv2.putText(frame, progress_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def _handle_keyboard_input(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('c'):
            self.event_queue.put(CaptureEvent())
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
        cv2.destroyAllWindows()


def layout_page(frames):
    """Layout function for printing"""
    # TODO: Implement actual layout logic
    logger.info(f"Laying out page with {len(frames)} frames")
    return None


if __name__ == "__main__":
    # Example with custom configuration
    config = PhotoboothConfig(
        photos_per_session=4,
        delay_between_photos=3.0,
        countdown_duration=2.0,
        gb_camera_config=GBCameraConfig(
            crop_start_x=100,
            crop_start_y=100
        ),
        nikon_config=NikonConfig(
            iso=800,
            image_format="JPEG"
        )
    )

    photobooth = Photobooth(config)
    photobooth.run()