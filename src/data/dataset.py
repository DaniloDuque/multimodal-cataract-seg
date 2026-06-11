import os
import json
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


## @brief Rasterizes COCO polygon segmentation annotations into a binary mask.
#  @param annotations  List of COCO annotation dicts for one image.
#  @param height       Image height in pixels.
#  @param width        Image width in pixels.
#  @return Binary mask (H, W) as float32.
def rasterize_masks(annotations: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for ann in annotations:
        for seg in ann.get("segmentation", []):
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1.0)
    return mask


## @brief PyTorch Dataset for the Roboflow Cataract-Seg dataset (COCO segmentation format).
#
#  Expects the dataset downloaded with format 'coco-segmentation':
#    <data_root>/{train,valid,test}/images/*.jpg
#    <data_root>/{train,valid,test}/_annotations.coco.json
#
#  Returns (rgb, edge, mask) tuples.
#
#  @param data_root Path to the dataset root directory.
#  @param split     One of 'train', 'valid', 'test'.
#  @param img_size  Target image size (square).
#  @param canny_t1  Canny lower threshold.
#  @param canny_t2  Canny upper threshold.
class CataractSegDataset(Dataset):
    def __init__(self, data_root: str, split: str, img_size: int = 512,
                 canny_t1: int = 50, canny_t2: int = 150):
        self.transform = get_transforms(img_size, split)
        self.canny_t1  = canny_t1
        self.canny_t2  = canny_t2

        ann_path = os.path.join(data_root, split, "_annotations.coco.json")
        with open(ann_path) as f:
            coco = json.load(f)

        # Determine where images actually live
        split_dir = os.path.join(data_root, split)
        img_subdir = os.path.join(split_dir, "images")
        self.img_dir = img_subdir if os.path.isdir(img_subdir) else split_dir

        # Build id → file_name map and id → annotations map
        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_image: dict[int, list] = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]
        file_name = os.path.basename(img_info["file_name"])
        img_path = os.path.join(self.img_dir, file_name)

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = rasterize_masks(
            self.anns_by_image[img_id],
            img_info["height"], img_info["width"]
        )

        augmented = self.transform(image=img, mask=mask)
        rgb  = augmented["image"].float()               # (3, H, W)
        mask = augmented["mask"].unsqueeze(0).float()   # (1, H, W)
        edge = canny_edge_map(rgb, self.canny_t1, self.canny_t2)  # (3, H, W)

        return rgb, edge, mask
