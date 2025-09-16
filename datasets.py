import cv2, torch, os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A


def resize_with_padding(img, size=256):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return img_padded


# -----------------------------
# Stage1 Dataset
# -----------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, img_root, mask_root, img_size=256, mask_suffix="_palpebral"):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.img_size = img_size
        self.mask_suffix = mask_suffix

        self.img_files, self.mask_files = [], []
        for img_path in self.img_root.rglob("*.jpg"):
            stem = img_path.stem
            mask_path = self.mask_root / f"{stem}{mask_suffix}.png"
            if mask_path.exists():
                self.img_files.append(str(img_path))
                self.mask_files.append(str(mask_path))

        print(f"[DEBUG] Found {len(self.img_files)} pairs under {img_root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_with_padding(img, self.img_size)

        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = resize_with_padding(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask


# -----------------------------
# Stage2 Dataset with Augmentation
# -----------------------------
class ConjAnemiaDataset(Dataset):
    def __init__(self, img_root, mask_root=None, mask_suffix="_mask",
                 img_size=256, augment=True,
                 exts=(".jpg", ".jpeg", ".png")):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root) if mask_root else self.img_root
        self.mask_suffix = mask_suffix
        self.exts = {e.lower() for e in exts}

        self.pairs = []
        for ext in self.exts:
            for img_path in self.img_root.rglob(f"*{ext}"):
                stem = img_path.stem
                mask_path = self.mask_root / f"{stem}{mask_suffix}.png"
                if mask_path.exists():
                    self.pairs.append((str(img_path), str(mask_path)))
                else:
                    print(f"[MISS] {mask_path} not found")

        self.transform = (
            A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.HueSaturationValue(p=0.15),
                A.RandomBrightnessContrast(p=0.15),
            ]) if augment else A.Compose([A.Resize(img_size, img_size)])
        )

        print(f"[DEBUG] Found {len(self.pairs)} pairs under {img_root} + {mask_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_fp, mask_fp = self.pairs[idx]
        img = cv2.imread(img_fp)
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)

        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=img, mask=mask)
        img, mask = aug["image"], aug["mask"]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)[None, ...]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

