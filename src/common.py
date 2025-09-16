"""
Common utilities and base classes for the photobooth project
"""
import cv2
import os
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class PhotoboothError(Exception):
    """Base exception for all photobooth errors"""

    def __init__(self, message: str, recoverable: bool = True, component: str = "Unknown"):
        super().__init__(message)
        self.recoverable = recoverable
        self.component = component

    def __str__(self) -> str:
        base_msg = super().__str__()
        parts = [f"[{self.component}] {base_msg}"]
        if not self.recoverable:
            parts.append("(Non-recoverable)")
        return " ".join(parts)


class Component(ABC):
    """Base class for all photobooth components"""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the component"""
        pass

    @abstractmethod
    def shutdown(self):
        """Cleanup component resources"""
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


@dataclass
class SystemStatus:
    """Overall system status"""
    cameras_ready: bool = False
    hardware_ready: bool = False
    printer_ready: bool = False
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def all_ready(self) -> bool:
        return self.cameras_ready and self.hardware_ready

    def add_error(self, error: str):
        self.error_count += 1
        self.last_error = error


class ImageFormat(Enum):
    """Supported image formats"""
    PNG = "png"
    JPEG = "jpg"
    BMP = "bmp"


@dataclass
class ImageMetadata:
    """Metadata for captured images"""
    timestamp: float = field(default_factory=time.time)
    camera_type: str = "unknown"
    session_id: Optional[str] = None
    image_index: Optional[int] = None
    processing_applied: list = field(default_factory=list)

    @property
    def formatted_timestamp(self) -> str:
        """Get human-readable timestamp"""
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.timestamp))


class PhotoboothImage:
    """
    Unified image class that handles both in-memory OpenCV arrays and file persistence
    """

    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 file_path: Optional[str] = None,
                 metadata: Optional[ImageMetadata] = None):
        """
        Initialize image from either numpy array or file path

        Args:
            data: OpenCV image data (np.ndarray)
            file_path: Path to image file
            metadata: Image metadata
        """
        self._data: Optional[np.ndarray] = data
        self._file_path: Optional[str] = file_path
        self.metadata = metadata or ImageMetadata()
        self._modified = False  # Track if in-memory data differs from file

        # Validation
        if data is None and file_path is None:
            raise ValueError("Either data or file_path must be provided")

    @property
    def data(self) -> np.ndarray:
        """Get image data as OpenCV array, loading from file if necessary"""
        if self._data is None and self._file_path:
            self._load_from_file()

        if self._data is None:
            raise ValueError("No image data available")

        return self._data

    @property
    def file_path(self) -> Optional[str]:
        """Get current file path"""
        return self._file_path

    @property
    def is_loaded(self) -> bool:
        """Check if image data is loaded in memory"""
        return self._data is not None

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get image shape if loaded"""
        return self.data.shape if self.is_loaded else None

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """Get image size as (width, height)"""
        if self.is_loaded:
            h, w = self.data.shape[:2]
            return (w, h)
        return None

    def _load_from_file(self):
        """Load image data from file"""
        if not self._file_path or not os.path.exists(self._file_path):
            raise FileNotFoundError(f"Image file not found: {self._file_path}")

        self._data = cv2.imread(self._file_path)
        if self._data is None:
            raise ValueError(f"Could not load image from {self._file_path}")

        logger.debug(f"Loaded image from {self._file_path}: {self._data.shape}")


    def save(self,
             file_path: Optional[str] = None,
             format: ImageFormat = ImageFormat.PNG,
             quality: int = 95) -> str:
        """
        Save image to file

        Args:
            file_path: Target file path (generates timestamp-based name if None)
            format: Image format
            quality: JPEG quality (0-100)

        Returns:
            Path where image was saved
        """
        if self._data is None:
            raise ValueError("No image data to save")

        # Generate file path if not provided
        if file_path is None:
            timestamp_ms = int(time.time() * 1000)
            camera_type = self.metadata.camera_type
            file_path = f"{camera_type}_{timestamp_ms}.{format.value}"

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Set quality for JPEG
        if format == ImageFormat.JPEG:
            success = cv2.imwrite(file_path, self._data, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            success = cv2.imwrite(file_path, self._data)

        if not success:
            raise IOError(f"Failed to save image to {file_path}")

        self._file_path = file_path
        self._modified = False
        logger.info(f"Image saved to {file_path}")
        return file_path

    def copy(self) -> 'PhotoboothImage':
        """Create a deep copy of the image"""
        new_data = self._data.copy() if self._data is not None else None
        new_metadata = ImageMetadata(
            timestamp=self.metadata.timestamp,
            camera_type=self.metadata.camera_type,
            session_id=self.metadata.session_id,
            image_index=self.metadata.image_index,
            processing_applied=self.metadata.processing_applied.copy()
        )

        return PhotoboothImage(data=new_data, file_path=self._file_path, metadata=new_metadata)

    def resize(self, size: Tuple[int, int], interpolation=cv2.INTER_AREA) -> 'PhotoboothImage':
        """
        Resize image and return new PhotoboothImage

        Args:
            size: Target size as (width, height)
            interpolation: OpenCV interpolation method
        """
        resized_data = cv2.resize(self.data, size, interpolation=interpolation)
        new_image = self.copy()
        new_image._data = resized_data
        new_image._modified = True
        new_image.metadata.processing_applied.append(f"resize_{size[0]}x{size[1]}")
        return new_image

    def crop(self, x: int, y: int, width: int, height: int) -> 'PhotoboothImage':
        """Crop image and return new PhotoboothImage"""
        cropped_data = self.data[y:y + height, x:x + width]
        new_image = self.copy()
        new_image._data = cropped_data
        new_image._modified = True
        new_image.metadata.processing_applied.append(f"crop_{x}_{y}_{width}_{height}")
        return new_image

    def apply_processing(self, processor_func, processor_name: str = "custom") -> 'PhotoboothImage':
        """
        Apply custom processing function and return new PhotoboothImage

        Args:
            processor_func: Function that takes and returns np.ndarray
            processor_name: Name for metadata tracking
        """
        processed_data = processor_func(self.data)
        new_image = self.copy()
        new_image._data = processed_data
        new_image._modified = True
        new_image.metadata.processing_applied.append(processor_name)
        return new_image

    def display(self, window_name: str = "Image", scale: float = 1.0):
        """Display image in OpenCV window"""
        display_data = self.data
        if scale != 1.0:
            display_data = cv2.resize(display_data, (0, 0), fx=scale, fy=scale)
        cv2.imshow(window_name, display_data)

    @classmethod
    def from_file(cls, file_path: str, metadata: Optional[ImageMetadata] = None) -> 'PhotoboothImage':
        """Create PhotoboothImage from file"""
        return cls(file_path=file_path, metadata=metadata)

    @classmethod
    def from_array(cls, data: np.ndarray, metadata: Optional[ImageMetadata] = None) -> 'PhotoboothImage':
        """Create PhotoboothImage from numpy array"""
        return cls(data=data, metadata=metadata)

    def __str__(self) -> str:
        shape_str = f"{self.shape}" if self.is_loaded else "not loaded"
        return f"PhotoboothImage({self.metadata.camera_type}, {shape_str}, {self._file_path})"


class ImageManager:
    """Utility class for managing collections of images and common operations"""

    def __init__(self, base_directory: str = "captures"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def create_session_directory(self, session_id: str) -> Path:
        """Create directory for a photo session"""
        session_dir = self.base_directory / session_id
        session_dir.mkdir(exist_ok=True)
        return session_dir

    def generate_filename(self, camera_type: str, session_id: str,
                          image_index: int, format: ImageFormat = ImageFormat.PNG) -> str:
        """Generate standardized filename"""
        timestamp = int(time.time() * 1000)
        return f"{camera_type}_{session_id}_{image_index:03d}_{timestamp}.{format.value}"

    def save_session_images(self, images: list, session_id: str) -> list:
        """Save all images in a session with consistent naming"""
        session_dir = self.create_session_directory(session_id)
        saved_paths = []

        for i, image in enumerate(images):
            if not isinstance(image, PhotoboothImage):
                raise TypeError("All images must be PhotoboothImage instances")

            filename = self.generate_filename(
                image.metadata.camera_type,
                session_id,
                i
            )
            file_path = session_dir / filename
            saved_path = image.save(str(file_path))
            saved_paths.append(saved_path)

        return saved_paths
