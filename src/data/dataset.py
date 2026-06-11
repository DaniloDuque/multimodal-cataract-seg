import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


## @brief Returns albumentations augmentation pipeline for the given split.
#  @param img_size Target image size (square).
#  @param split    One of 'train', 'valid', 'test'.
#  @return albumentations.Compose pipeline.
def get_transforms(img_size: int, split: str) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


## @brief Computes a 3-channel Canny edge map from an RGB image tensor.
#  @param img_tensor Float tensor (3, H, W) in [0, 1].
#  @param t1        Canny lower threshold.
#  @param t2        Canny upper threshold.
#  @return Float tensor (3, H, W) edge map in [0, 1].
def canny_edge_map(img_tensor: torch.Tensor, t1: int = 50, t2: int = 150) -> torch.Tensor:
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, t1, t2).astype(np.float32) / 255.0
    return torch.from_numpy(edges).unsqueeze(0).repeat(3, 1, 1)


## @brief PyTorch Dataset for the Roboflow Cataract-Seg dataset.
#
#  Expects the dataset downloaded in Semantic Segmentation (PNG masks) format:
#    <data_root>/{train,valid,test}/images/*.jpg
#    <data_root>/{train,valid,test}/masks/*.png
#
#  Returns (rgb, edge, mask) tuples. Baselines that don't use the edge stream
#  can simply ignore it.
#
#  @param data_root Path to the dataset root directory.
#  @param split     One of 'train', 'valid', 'test'.
#  @param img_size  Target image size (square).
#  @param canny_t1  Canny lower threshold.
#  @param canny_t2  Canny upper threshold.
class CataractSegDataset(Dataset):
    def __init__(self, data_root: str, split: str, img_size: int = 512,
                 canny_t1: int = 50, canny_t2: int = 150):
        self.img_dir   = os.path.join(data_root, split, "images")
        self.mask_dir  = os.path.join(data_root, split, "masks")
        self.transform = get_transforms(img_size, split)
        self.canny_t1  = canny_t1
        self.canny_t2  = canny_t2
        self.ids       = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name  = os.path.splitext(self.ids[idx])[0]
        img   = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, self.ids[idx])),
                              cv2.COLOR_BGR2RGB)
        mask_path = os.path.join(self.mask_dir, name + ".png")
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 0).astype(np.float32)

        augmented = self.transform(image=img, mask=mask)
        rgb  = augmented["image"].float()          # (3, H, W)
        mask = augmented["mask"].unsqueeze(0).float()  # (1, H, W)
        edge = canny_edge_map(rgb, self.canny_t1, self.canny_t2)  # (3, H, W)

        return rgb, edge, mask
