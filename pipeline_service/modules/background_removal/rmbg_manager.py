from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop
from config import Settings
from logger_config import logger

from ben2 import BEN_Base

class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings

        # Set padding percentage and output size for centering and resizing
        self.padding_percentage = self.settings.padding_percentage
        self.limit_padding = self.settings.limit_padding
        self.output_size = self.settings.output_image_size

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # Set BEN model
        self.model: BEN_Base | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.background_removal_model_id} model...")

        # Load model
        try:
            self.model = BEN_Base.from_pretrained(self.settings.background_removal_model_id).to(self.device).eval()
            logger.success(f"{self.settings.background_removal_model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.background_removal_model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.background_removal_model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.background_removal_model_id} model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        # try:
        t1 = time.time()
        # Check if the image has alpha channel
        has_alpha = False
        
        if image.mode == "RGBA":
            # Get alpha channel
            alpha = np.array(image)[:, :, 3]
            if not np.all(alpha==255):
                has_alpha=True
        
        if has_alpha:
            # If the image has alpha channel, return the image
            output = image
            
        else:
            # PIL.Image (H, W, C) C=3
            rgb_image = image.convert('RGB').resize(self.settings.input_image_size)
            
            # Tensor (H, W, C) -> (C, H',W')
            # rgb_tensor = self.transforms(rgb_image).to(self.device)
            foreground_tensor = self._remove_background(rgb_image)
            output = self._crop_and_center(foreground_tensor)

            output = image_without_background = to_pil_image(output[:3])

        removal_time = time.time() - t1
        logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {output.size} - InputSize: {image.size}")

        return image_without_background
            
        # except Exception as e:
        #     logger.error(f"Error removing background: {e}")
        #     return image 

    def _remove_background(self, image: Image) -> torch.Tensor:
        """
        Remove the background from the image.
        """
        with torch.no_grad():
            foreground = self.model.inference(image.copy())
        return self.transforms(foreground)

    def _crop_and_center(self, foreground_tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """

        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]

        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > 0.8)
        logger.info(f"BBOX len: {len(bbox_indices)}")
        if len(bbox_indices) == 0:
            crop_args = dict(top = 0, left = 0, height = mask.shape[1], width = mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center =  (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = int(size * padded_size_factor)

            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)
            
            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )
        

        logger.info(f"CROP: {crop_args}")
        mask = mask.unsqueeze(0)
        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([tensor_rgb*mask, mask], dim=-3)
        output = resized_crop(tensor_rgba, **crop_args, size = self.output_size, antialias=False)
        return output