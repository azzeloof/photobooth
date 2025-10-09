import os
import time
import logging
import random
import requests
import replicate
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from common import PhotoboothImage, ImageMetadata

logger = logging.getLogger(__name__)


class AIProcessor:
    """Handles AI upscaling operations using Replicate API"""

    def __init__(self, api_token: str, deployment_name: str = "satrat/gameboy-upsample"):
        os.environ["REPLICATE_API_TOKEN"] = api_token
        self.deployment_name = deployment_name
        self.deployment = None
        self._ai_border_image = None
        self._initialize_deployment()
        self._load_ai_border_image()

    def _initialize_deployment(self):
        """Initialize the Replicate deployment"""
        try:
            self.deployment = replicate.deployments.get(self.deployment_name)
            logger.info(f"Initialized Replicate deployment: {self.deployment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize deployment {self.deployment_name}: {e}")
            self.deployment = None

    def _load_ai_border_image(self):
        """Load the AI border image"""
        try:
            self._ai_border_image = cv2.imread("gb_ai_border.png")
            if self._ai_border_image is not None:
                logger.info(f"AI border image loaded: {self._ai_border_image.shape}")
            else:
                logger.warning("Could not load gb_ai_border.png")
        except Exception as e:
            logger.error(f"Error loading AI border image: {e}")

    def process_image_sync(self, image: PhotoboothImage,
                           prompt: str = "high quality colorized photograph, natural colors, detailed",
                           negative_prompt: str = "blurry, low quality",
                           num_inference_steps: int = 20,
                           cfg_scale: float = 7.5,
                           timeout: int = 30) -> PhotoboothImage:
        """
        Process a single image through the AI upscaling API (synchronous version)

        Args:
            image: PhotoboothImage to process
            prompt: AI processing prompt
            negative_prompt: Negative prompt for AI
            num_inference_steps: Number of inference steps
            cfg_scale: CFG scale parameter

        Returns:
            AI-processed PhotoboothImage with border
        """
        if not self.deployment:
            logger.error("Deployment not initialized, falling back to original image")
            return image.copy()

        try:
            # Save temporary file if needed
            temp_path = None
            if not image.file_path:
                timestamp_ms = int(time.time() * 1000)
                temp_path = f"temp_ai_input_{timestamp_ms}.png"
                image.save(temp_path)
                input_path = temp_path
            else:
                input_path = image.file_path

            # Run inference using deployment
            logger.info(f"Starting AI processing for {input_path}")

            with open(input_path, "rb") as f:
                prediction = self.deployment.predictions.create(
                    input={
                        "image": f,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "steps": num_inference_steps,
                        "cfg_scale": cfg_scale,
                        "seed": random.randint(0, 999999) #42  # for reproducibility
                    }
                )

            # Wait for the prediction to complete
            logger.info("Waiting for AI prediction to complete...")
            #prediction.wait(timeout=timeout)
            start_time = time.time()
            prediction.reload()
            while prediction.status not in ['succeeded', 'failed', 'canceled']:
                if time.time() - start_time > timeout:
                    logger.error(f"AI prediction timed out after {timeout} seconds for image {image.file_path}")
                    try:
                        prediction.cancel()
                        logger.info("Cancelled the timed-out Replicate prediction")
                    except Exception as e:
                        logger.warning(f"Could not cancel prediction: {e}")
                    return image.copy()
                time.sleep(1)
                prediction.reload()

            if prediction.status == "succeeded":
                output_url = prediction.output
                logger.info(f"AI prediction succeeded, downloading result from: {output_url}")
            else:
                logger.error(f"AI prediction failed with status: {prediction.status}")
                if hasattr(prediction, 'error') and prediction.error:
                    logger.error(f"Prediction error: {prediction.error}")
                return image.copy()

            # Download result
            ai_image = self._download_result(output_url)

            # Convert PIL image to OpenCV format
            ai_image_cv = cv2.cvtColor(np.array(ai_image), cv2.COLOR_RGB2BGR)

            # Add AI border frame
            ai_image_with_border = self._add_ai_border(ai_image_cv)

            # Create AI-processed PhotoboothImage with border
            ai_metadata = ImageMetadata(
                timestamp=image.metadata.timestamp,
                camera_type="gameboy_ai_framed",
                session_id=image.metadata.session_id,
                image_index=image.metadata.image_index,
                processing_applied=image.metadata.processing_applied.copy() + ["ai_upscale", "ai_border"]
            )

            ai_photobooth_image = PhotoboothImage.from_array(ai_image_with_border, ai_metadata)

            # Save AI result
            timestamp_ms = int(image.metadata.timestamp * 1000)
            ai_filepath = os.path.join("captures", "gameboy", f"gameboy_ai_framed_{timestamp_ms}.png")
            ai_photobooth_image.save(ai_filepath)

            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            logger.info(f"AI processing completed: {ai_filepath}")
            return ai_photobooth_image
        #except TimeoutError:
        #    logger.error(f"AI prediction timed out after {timeout} seconds for image {image.file_path}")
        #    return image.copy()
        except Exception as e:
            logger.error(f"AI processing failed: {e}")
            # Return original image as fallback
            return image.copy()

    def _add_ai_border(self, ai_image: np.ndarray) -> np.ndarray:
        """Add the AI border frame to the AI-processed image"""
        if self._ai_border_image is None:
            logger.warning("No AI border image available, returning AI image as-is")
            return ai_image

        try:
            # Create a copy of the border to avoid modifying the cached version
            bordered_frame = self._ai_border_image.copy()

            # The AI border has different dimensions and offset than the GB border
            # Based on the fallback method, it looks like the offset is (96, 96)
            border_offset_x, border_offset_y = 96, 96

            # Resize AI image to fit in the border if needed
            target_height = bordered_frame.shape[0] - (2 * border_offset_y)
            target_width = bordered_frame.shape[1] - (2 * border_offset_x)
            
            if ai_image.shape[:2] != (target_height, target_width):
                ai_image_resized = cv2.resize(ai_image, (target_width, target_height))
                logger.info(f"Resized AI image from {ai_image.shape[:2]} to {ai_image_resized.shape[:2]} to fit border")
            else:
                ai_image_resized = ai_image

            # Insert the AI image into the border
            bordered_frame[border_offset_y:border_offset_y + ai_image_resized.shape[0],
                          border_offset_x:border_offset_x + ai_image_resized.shape[1]] = ai_image_resized

            return bordered_frame

        except Exception as error:
            logger.error(f"AI border addition failed: {error}")
            return ai_image  # Return without border if it fails

    def _download_result(self, output_url: str) -> Image.Image:
        """Download the result image"""
        try:
            response = requests.get(output_url, timeout=30)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                raise Exception(f"Failed to download result: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading result: {e}")
            raise