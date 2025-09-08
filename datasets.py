import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A


class BaseConjDataset(Dataset):
    """Load image/mask pairs, supporting optional mask suffix and augmentation."""

    def __init__(self, img_root, mask_root=None, mask_suffix="", img_size=256,
                 augment=False, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root) if mask_root else self.img_root
        self.mask_suffix = mask_suffix
        self.exts = {e.lower() for e in exts}

        self.pairs = []
        for img_path in self.img_root.iterdir():
            if img_path.suffix.lower() in self.exts:
                stem = img_path.stem
                mask_path = self.mask_root / f"{stem}{mask_suffix}.png"
                if mask_path.exists():
                    self.pairs.append((str(img_path), str(mask_path)))

        self.transform = (
            A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.HueSaturationValue(p=0.15),
                A.RandomBrightnessContrast(p=0.15),
            ]) if augment else A.Compose([A.Resize(img_size, img_size)])
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_fp, mask_fp = self.pairs[idx]
        img = cv2.imread(img_fp)
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)

        # --- 크기 맞추기 ---
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # -------------------

        aug = self.transform(image=img, mask=mask)
        img, mask = aug["image"], aug["mask"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
        mask = (mask > 127).astype(np.float32)[None, ...]
        return img, mask


class ConjAnemiaDataset(Dataset):
    """Dataset that recursively scans folders for image/mask pairs.

    This version searches all sub-directories under ``img_root`` using
    ``Path.rglob("*.jpg")`` (and other extensions) so that images can be
    organised in arbitrary nested structures.  Masks are expected to have the
    same relative path under ``mask_root`` and use ``mask_suffix`` before the
    ``.png`` extension.
    """

    def __init__(self, img_root, mask_root=None, mask_suffix="", img_size=256,
                 augment=False, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root) if mask_root else self.img_root
        self.mask_suffix = mask_suffix
        self.exts = {e.lower() for e in exts}

        self.pairs = []
        for ext in self.exts:
            for img_path in self.img_root.rglob(f"*{ext}"):
                rel = img_path.relative_to(self.img_root)
                mask_path = self.mask_root / rel.parent / f"{img_path.stem}{mask_suffix}.png"
                if mask_path.exists():
                    self.pairs.append((str(img_path), str(mask_path)))

        self.transform = (
            A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.HueSaturationValue(p=0.15),
                A.RandomBrightnessContrast(p=0.15),
            ]) if augment else A.Compose([A.Resize(img_size, img_size)])
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_fp, mask_fp = self.pairs[idx]
        img = cv2.imread(img_fp)
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)

        # --- 크기 맞추기 ---
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # -------------------

        aug = self.transform(image=img, mask=mask)
        img, mask = aug["image"], aug["mask"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)[None, ...]

        return img, mask
