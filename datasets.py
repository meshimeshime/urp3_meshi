import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset

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
        cv2.BORDER_CONSTANT, value=0
    )
    return img_padded


class ImageFolderDataset(Dataset):
    def __init__(self, img_root, mask_root, img_size=256):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.img_size = img_size

        self.img_files, self.mask_files = [], []

        # jpg 파일을 기준으로 대응되는 _palpebral.png 찾기
        for img_path in self.img_root.rglob("*.jpg"):
            stem = img_path.stem
            mask_path = img_path.with_name(f"{stem}_palpebral.png")

            # _forniceal_palpebral 같은 건 무시
            if mask_path.exists() and "forniceal" not in mask_path.name:
                self.img_files.append(str(img_path))
                self.mask_files.append(str(mask_path))

        print(f"[DEBUG] Found {len(self.img_files)} pairs under {img_root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 입력 이미지 (RGB)
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_with_padding(img, self.img_size)

        # 정답 mask (흑백)
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = resize_with_padding(mask, self.img_size)
        mask = (mask > 127).astype("float32")

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask
