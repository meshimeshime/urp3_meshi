import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Simple loader for the Conjunctival Images for Anemia Detection dataset.

    It searches for ``*.jpg`` images under ``img_root`` and expects a
    corresponding mask with ``{stem}{mask_suffix}.png`` alongside each image.
    """

    def __init__(self, img_root, mask_root, mask_suffix="", img_size=256):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.mask_suffix = mask_suffix
        self.img_size = img_size

        self.img_files, self.mask_files = [], []
        for img_path in self.img_root.rglob("*.jpg"):
            mask_path = img_path.with_name(f"{img_path.stem}{mask_suffix}.png")
            if mask_path.exists():
                self.img_files.append(str(img_path))
                self.mask_files.append(str(mask_path))

        print(f"[DEBUG] Found {len(self.img_files)} images with masks under {self.img_root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = mask.repeat(3, 1, 1)  # 흑백 마스크를 3채널로 확장

        return img, mask
