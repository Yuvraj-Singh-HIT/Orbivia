import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

CLASS_MAP = {
    0: 0,    # Trees
    1: 1,    # Lush Bushes  
    2: 2,    # Dry Grass
    3: 3,    # Dry Bushes
    27: 4,   # Ground Clutter
    39: 5,   # Flowers
    # Keep original mappings for compatibility
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9,
}


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = sorted(os.listdir(img_dir))
        if mask_dir is not None:
            self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.resize(image, (self.size, self.size))
        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            raw_mask = cv2.imread(mask_path, 0)
            if raw_mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")

            raw_mask = cv2.resize(
                raw_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST
            )

            clean_mask = np.zeros_like(raw_mask, dtype=np.uint8)

            for k, v in CLASS_MAP.items():
                clean_mask[raw_mask == k] = v

            mask = torch.tensor(clean_mask).long()
            return image, mask

        return image
