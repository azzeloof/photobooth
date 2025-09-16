import gphoto2 as gp
import os
import signal
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
from contextlib import contextmanager
from common import PhotoboothError, PhotoboothImage, ImageManager, ImageMetadata

logger = logging.getLogger(__name__)

@dataclass
class NikonConfig:
    """Configuration for Nikon camera operations"""
    capture_directory: str = "captures/nikon"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: float = 30.0
    # Camera settings (can be expanded)
    iso: Optional[int] = None
    aperture: Optional[str] = None
    shutter_speed: Optional[str] = None
    image_format: str = "JPEG"  # or "RAW", "RAW+JPEG"


class NikonError(PhotoboothError):
    """Custom exception for Nikon camera errors"""
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message, recoverable, component="NikonCamera")


class NikonCamera:
    """
    Enhanced Nikon camera interface with robust error handling and configuration
    """

    def __init__(self, config: Optional[NikonConfig] = None):
        self.config = config or NikonConfig()
        self.camera: Optional[gp.Camera] = None
        self.camera_lock = threading.RLock()
        self.image_manager = ImageManager(self.config.capture_directory)
        self._is_initialized = False
        self._camera_info: Dict[str, Any] = {}

        self._ensure_capture_directory()
        self._initialize_camera()

    def _ensure_capture_directory(self):
        """Ensure the capture directory exists"""
        try:
            Path(self.config.capture_directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Capture directory ready: {self.config.capture_directory}")
        except Exception as e:
            raise NikonError(f"Could not create capture directory: {e}")

    def _kill_conflicting_processes(self):
        """
        Finds and kills processes known to conflict with gphoto2 on Linux.
        This uses `pgrep` to find processes by name.
        """
        conflicting_processes = ['gvfs-gphoto2-volume-monitor', 'gvfsd-gphoto2']
        logger.info("Attempting to find and kill conflicting processes...")

        try:
            for process_name in conflicting_processes:
                # Use pgrep to find the PID of the conflicting process
                result = subprocess.run(['pgrep', '-f', process_name], capture_output=True, text=True)
                if result.stdout:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid:
                            try:
                                logger.warning(
                                    f"Found conflicting process '{process_name}' with PID {pid}. Terminating it...")
                                os.kill(int(pid), signal.SIGTERM)
                                # Wait a moment for the process to be terminated
                                time.sleep(0.5)
                            except (ValueError, ProcessLookupError, PermissionError) as e:
                                logger.error(f"Failed to kill process {pid}: {e}")
        except FileNotFoundError:
            logger.warning("'pgrep' command not found. Cannot automatically kill conflicting processes.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while trying to kill processes: {e}")

    def _initialize_camera(self):
        """Initialize the camera with comprehensive error handling"""
        max_init_retries = 2  # Try once, kill process, then try again
        for attempt in range(max_init_retries):
            try:
                # Check if gPhoto2 can detect any cameras
                camera_list = gp.check_result(gp.gp_camera_autodetect())
                if not camera_list:
                    raise NikonError("No cameras detected by gPhoto2")

                logger.info(f"Detected cameras: {[cam[0] for cam in camera_list]}")

                # Initialize camera
                self.camera = gp.Camera()
                self.camera.init()

                # Get camera information
                self._get_camera_info()

                # Apply initial configuration
                self._apply_camera_settings()

                self._is_initialized = True
                logger.info(f"Camera initialized successfully: {self._camera_info.get('model', 'Unknown')}")
                # If initialization is successful, break out of the retry loop
                break

            except gp.GPhoto2Error as error:
                # If the camera is busy on the first attempt, try to kill the conflicting process
                if error.code == gp.GP_ERROR_IO_USB_CLAIM and attempt < max_init_retries - 1:
                    logger.warning("Camera is busy. Attempting to resolve conflict automatically...")
                    self._kill_conflicting_processes()
                    # Wait a second before retrying
                    time.sleep(1)
                    continue  # Continue to the next attempt in the loop

                self.camera = None
                if error.code == gp.GP_ERROR_MODEL_NOT_FOUND:
                    raise NikonError(
                        "Camera not found. Please ensure your camera is connected, turned on, and in the correct mode (not in sleep mode)")
                elif error.code == gp.GP_ERROR_IO_USB_CLAIM:
                    raise NikonError("Camera is busy or already in use by another application")
                elif error.code == gp.GP_ERROR_TIMEOUT:
                    raise NikonError("Camera connection timed out. Check USB connection and camera power")
                else:
                    raise NikonError(f"Camera initialization failed: {error} (code: {error.code})")
            except Exception as error:
                self.camera = None
                raise NikonError(f"Unexpected error during camera initialization: {error}")

    #def _create_timeout_config(self) -> gp.CameraWidget:
    #    """Create timeout configuration for camera operations"""
    #    try:
    #        config = self.camera.get_config()
    #        # Set timeout if available (implementation depends on the camera model)
    #        return config
    #    except:
    #        # Return empty config if the timeout setting fails
    #        return gp.CameraWidget()

    def _get_camera_info(self):
        """Retrieve camera information and capabilities"""
        if not self.camera:
            return

        try:
            # Get camera summary
            summary = self.camera.get_summary()
            self._camera_info['summary'] = str(summary)

            # Try to get model information
            try:
                config = self.camera.get_config()
                model_widget = config.get_child_by_name('cameramodel')
                self._camera_info['model'] = model_widget.get_value()
            except:
                self._camera_info['model'] = 'Unknown Model'

            # Get available capture formats
            try:
                config = self.camera.get_config()
                format_widget = config.get_child_by_name('imageformat')
                choices = [format_widget.get_choice(i) for i in range(format_widget.count_choices())]
                self._camera_info['formats'] = choices
            except:
                self._camera_info['formats'] = []

        except Exception as error:
            logger.warning(f"Could not retrieve camera info: {error}")

    def _apply_camera_settings(self):
        """Apply configured camera settings"""
        if not self.camera:
            return

        settings_applied = []

        try:
            config = self.camera.get_config()

            # Set ISO if configured
            if self.config.iso:
                try:
                    iso_widget = config.get_child_by_name('iso')
                    iso_widget.set_value(str(self.config.iso))
                    settings_applied.append(f"ISO: {self.config.iso}")
                except:
                    logger.warning(f"Could not set ISO to {self.config.iso}")

            # Set aperture if configured
            if self.config.aperture:
                try:
                    aperture_widget = config.get_child_by_name('f-number')
                    aperture_widget.set_value(self.config.aperture)
                    settings_applied.append(f"Aperture: {self.config.aperture}")
                except:
                    logger.warning(f"Could not set aperture to {self.config.aperture}")

            # Set shutter speed if configured
            if self.config.shutter_speed:
                try:
                    shutter_widget = config.get_child_by_name('shutterspeed')
                    shutter_widget.set_value(self.config.shutter_speed)
                    settings_applied.append(f"Shutter: {self.config.shutter_speed}")
                except:
                    logger.warning(f"Could not set shutter speed to {self.config.shutter_speed}")

            # Set the image format if configured
            try:
                format_widget = config.get_child_by_name('imageformat')
                if self.config.image_format in self._camera_info.get('formats', []):
                    format_widget.set_value(self.config.image_format)
                    settings_applied.append(f"Format: {self.config.image_format}")
            except:
                logger.warning(f"Could not set image format to {self.config.image_format}")

            # Apply all settings
            if settings_applied:
                self.camera.set_config(config)
                logger.info(f"Camera settings applied: {', '.join(settings_applied)}")

        except Exception as e:
            logger.warning(f"Error applying camera settings: {e}")

    @contextmanager
    def _camera_operation(self, operation_name: str):
        """Context manager for camera operations with proper locking and error handling"""
        if not self._is_initialized:
            raise NikonError("Camera not initialized")

        with self.camera_lock:
            try:
                logger.debug(f"Starting camera operation: {operation_name}")
                yield
                logger.debug(f"Completed camera operation: {operation_name}")
            except gp.GPhoto2Error as e:
                logger.error(f"gPhoto2 error during {operation_name}: {e}")
                if e.code == gp.GP_ERROR_CAMERA_BUSY:
                    raise NikonError(f"Camera is busy during {operation_name}")
                elif e.code == gp.GP_ERROR_TIMEOUT:
                    raise NikonError(f"Camera operation {operation_name} timed out")
                elif e.code == gp.GP_ERROR_IO:
                    raise NikonError(f"Communication error during {operation_name}")
                else:
                    raise NikonError(f"{operation_name} failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during {operation_name}: {e}")
                raise NikonError(f"Unexpected error during {operation_name}: {e}")

    def capture_image(self, session_id: Optional[str] = None) -> PhotoboothImage:
        """
        Capture photo and return as PhotoboothImage

        Returns:
            PhotoboothImage with captured data
        """
        # Use existing capture logic but return PhotoboothImage
        file_path = self.capture_to_file()  # Rename existing capture method

        if file_path is None:
            raise NikonError("Capture failed")

        # Create metadata
        metadata = ImageMetadata(
            camera_type="nikon",
            session_id=session_id,
            timestamp=time.time()
        )

        return PhotoboothImage.from_file(file_path, metadata)

    def capture_to_file(self, custom_filename: Optional[str] = None) -> Optional[str]:
        """
        Capture a photo with the Nikon camera

        Args:
            custom_filename: Optional custom filename (without extension)

        Returns:
            Path to the captured image file, or None if capture failed

        Raises:
            NikonError: If capture operation fails
        """
        if not self._is_initialized:
            raise NikonError("Camera not initialized")

        for attempt in range(self.config.max_retries):
            try:
                with self._camera_operation("capture"):
                    # Trigger the capture
                    logger.info(f"Capturing image (attempt {attempt + 1}/{self.config.max_retries})")
                    file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)

                    # Generate target filename
                    if custom_filename:
                        # Use custom filename but keep the original extension
                        original_name, ext = os.path.splitext(file_path.name)
                        target_filename = f"{custom_filename}{ext}"
                    else:
                        # Use timestamp-based filename
                        timestamp = int(time.time() * 1000)
                        original_name, ext = os.path.splitext(file_path.name)
                        target_filename = f"nikon_{timestamp}{ext}"

                    target_path = os.path.join(self.config.capture_directory, target_filename)

                    # Retrieve the file from camera
                    logger.debug(f"Downloading image: {file_path.name} -> {target_path}")
                    camera_file = self.camera.file_get(
                        file_path.folder,
                        file_path.name,
                        gp.GP_FILE_TYPE_NORMAL
                    )

                    # Save to local filesystem
                    camera_file.save(target_path)

                    # Verify the file was saved correctly
                    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                        raise NikonError("Captured file is missing or empty")

                    logger.info(f"Image captured successfully: {target_path}")
                    return target_path

            except NikonError:
                # Re-raise NikonError as-is
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Capture attempt {attempt + 1} failed, retrying...")
                time.sleep(self.config.retry_delay)

            except Exception as e:
                error_msg = f"Unexpected error during capture: {e}"
                if attempt == self.config.max_retries - 1:
                    raise NikonError(error_msg)
                logger.warning(f"{error_msg}, retrying...")
                time.sleep(self.config.retry_delay)
        return None

    # Backwards compatibility
    def capture(self, custom_filename: Optional[str] = None) -> Optional[str]:
        """Backwards compatibility - returns file path"""
        return self.capture_to_file(custom_filename)

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and status"""
        return {
            'initialized': self._is_initialized,
            'model': self._camera_info.get('model', 'Unknown'),
            'available_formats': self._camera_info.get('formats', []),
            'config': {
                'iso': self.config.iso,
                'aperture': self.config.aperture,
                'shutter_speed': self.config.shutter_speed,
                'image_format': self.config.image_format
            }
        }

    def test_connection(self) -> bool:
        """Test if a camera connection is working"""
        if not self._is_initialized:
            return False

        try:
            with self._camera_operation("connection_test"):
                # Try to get the camera summary as a connection test
                summary = self.camera.get_summary()
                return True
        except:
            return False

    def release(self):
        """Release camera resources"""
        logger.info("Releasing camera resources...")

        with self.camera_lock:
            if self.camera:
                try:
                    self.camera.exit()
                    logger.info("Camera released successfully")
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                finally:
                    self.camera = None
                    self._is_initialized = False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    @property
    def is_initialized(self) -> bool:
        """Check if a camera is initialized and ready"""
        return self._is_initialized

    @property
    def camera_model(self) -> str:
        """Get camera model name"""
        return self._camera_info.get('model', 'Unknown')


# Backwards compatibility alias
Nikon = NikonCamera

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with custom configuration
    config = NikonConfig(
        capture_directory="test_captures",
        iso=800,
        image_format="JPEG"
    )

    try:
        with NikonCamera(config) as camera:
            print(f"Camera Info: {camera.get_camera_info()}")

            # Test connection
            if camera.test_connection():
                print("Camera connection test: PASSED")

                # Capture test image
                print("Capturing test image...")
                image_path = camera.capture("test_image")
                if image_path:
                    print(f"Test image captured: {image_path}")
                else:
                    print("Test capture failed")
            else:
                print("Camera connection test: FAILED")

    except NikonError as error:
        print(f"Nikon camera error: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")