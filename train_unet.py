# -*- coding: utf-8 -*-
"""
U-Net training on CP-AnemiC ROI images with auto-generated GT masks.
입력:  .\CP-AnemiC dataset\(Anemic|Non-anemic)\Image_XXX.png
마스크: .\cp_gt_masks\Image_XXX_mask.png  (없으면 스킵)
출력:  models/unet_cp_best.pt, logs, 예측 시각화 preds/
"""

from __future__ import annotations
import argparse
import os, random, math, time
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Config
# ----------------------------
class Cfg:
    root = Path("./CP-AnemiC dataset")
    mask_root = Path("./cp_gt_masks")
    out_dir = Path("./models")
    pred_dir = Path("./preds")
    img_size: int = 512         # 짧은 변 기준으로 리사이즈 후 중앙 crop/pad
    batch: int = 6
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-5
    val_ratio: float = 0.15
    num_workers: int = 4
    seed: int = 42
    amp: bool = True            # 자동 혼합정밀
    save_every_pred: int = 3    # N에폭마다 예측 샘플 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on CP-AnemiC ROI images")
    parser.add_argument("--root", type=Path, default=Cfg.root, help="dataset root directory")
    parser.add_argument("--mask-root", type=Path, default=Cfg.mask_root, help="ground truth mask directory")
    parser.add_argument("--epochs", type=int, default=Cfg.epochs, help="number of training epochs")
    parser.add_argument("--batch", type=int, default=Cfg.batch, help="batch size")
    parser.add_argument("--lr", type=float, default=Cfg.lr, help="learning rate")
    parser.add_argument("--img-size", type=int, default=Cfg.img_size, help="input image size")
    parser.add_argument("--device", default=Cfg.device, help="training device")
    return parser.parse_args()

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def list_images(root:Path, exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def resize_letterbox(img:np.ndarray, size:int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    h, w = img.shape[:2]
    scale = size / min(h, w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # 중앙 크롭(또는 패드)
    top = max((nh - size)//2, 0)
    left = max((nw - size)//2, 0)
    img_c = img_rs[top:top+size, left:left+size]
    # 부족하면 패딩
    if img_c.shape[0] != size or img_c.shape[1] != size:
        out = np.zeros((size, size, img_c.shape[2]), img_c.dtype) if img_c.ndim==3 else np.zeros((size,size), img_c.dtype)
        out[:img_c.shape[0], :img_c.shape[1]] = img_c
        img_c = out
    return img_c, (top,left,nh,nw)

def to_tensor(img:np.ndarray) -> torch.Tensor:
    if img.ndim==2:  # H,W
        img = img[None,...]
    else:            # H,W,C -> C,H,W
        img = img.transpose(2,0,1)
    img = img.astype(np.float32)/255.0
    return torch.from_numpy(img)

# ----------------------------
# Dataset
# ----------------------------
class CPConjDataset(Dataset):
    def __init__(self, root:Path, mask_root:Path, img_size:int=512, items:List[Path]|None=None, augment:bool=False):
        self.root = root; self.mask_root = mask_root
        self.img_size = img_size; self.augment = augment
        if items is None:
            imgs = list_images(root)
        else:
            imgs = items
        valid = []
        for ip in imgs:
            m = mask_root / f"{ip.stem}_mask.png"
            if m.exists():
                valid.append((ip, m))
        if not valid:
            raise FileNotFoundError("No (image, mask) pairs. Check paths and suffix(_mask).")
        self.items = valid

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None: raise RuntimeError(f"fail image: {ip}")
        if mask is None: raise RuntimeError(f"fail mask: {mp}")

        img, _ = resize_letterbox(img, self.img_size)
        mask, _ = resize_letterbox(mask, self.img_size)

        if self.augment:
            if random.random()<0.5:
                img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)
            if random.random()<0.15:
                a = 1.0 + (random.random()-0.5)*0.4
                b = (random.random()-0.5)*40
                img = np.clip(img*a + b, 0, 255).astype(np.uint8)

        mask = (mask>127).astype(np.float32)
        img_t = to_tensor(img)
        mask_t = torch.from_numpy(mask)[None, ...].float()
        return img_t, mask_t, str(ip)

# ----------------------------
# Model
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8)
        self.p4 = nn.MaxPool2d(2)

        self.bott = DoubleConv(base*8, base*16)

        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.up4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.up3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.up2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.up1 = DoubleConv(base*2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        xb = self.bott(self.p4(x4))

        x = self.u4(xb); x = torch.cat([x4, x], dim=1); x = self.up4(x)
        x = self.u3(x);  x = torch.cat([x3, x], dim=1); x = self.up3(x)
        x = self.u2(x);  x = torch.cat([x2, x], dim=1); x = self.up2(x)
        x = self.u1(x);  x = torch.cat([x1, x], dim=1); x = self.up1(x)
        return self.outc(x)

# ----------------------------
# Loss & Metrics
# ----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps:float=1e-6): super().__init__(); self.eps=eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2*(probs*targets).sum(dim=(2,3)) + self.eps
        den = (probs*probs).sum(dim=(2,3)) + (targets*targets).sum(dim=(2,3)) + self.eps
        dice = num/den
        return 1 - dice.mean()

def iou_score(logits, targets, thr:float=0.5, eps:float=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs>thr).float()
    inter = (preds*targets).sum(dim=(2,3))
    union = (preds+targets - preds*targets).sum(dim=(2,3)) + eps
    return (inter/union).mean().item()

# ----------------------------
# Train
# ----------------------------
def split_train_val(items:List[Path], val_ratio:float, seed:int=42):
    rng = random.Random(seed); rng.shuffle(items)
    n = len(items); v = int(n*val_ratio)
    return items[v:], items[:v]

def save_preds(model, loader, device, out_dir:Path, max_save:int=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved=0
    with torch.no_grad():
        for imgs, masks, names in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(names)):
                if saved>=max_save: return
                im = (imgs[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
                pr = (probs[i,0]>0.5).astype(np.uint8)*255
                roi = im.copy()
                roi[pr==0] = 0
                ov = im.copy()
                red = np.zeros_like(im); red[:,:,2] = 255
                ov[pr>0] = cv2.addWeighted(im, 0.6, red, 0.4, 0)[pr>0]
                stem = Path(names[i]).stem
                cv2.imwrite(str(out_dir/f"{stem}_pred.png"), pr)
                cv2.imwrite(str(out_dir/f"{stem}_roi.png"), roi)
                cv2.imwrite(str(out_dir/f"{stem}_overlay.png"), ov)
                saved+=1

def train(args):
    device = args.device
    set_seed(Cfg.seed)
    all_imgs = list_images(args.root)
    tr_items, va_items = split_train_val(all_imgs, Cfg.val_ratio, Cfg.seed)
    train_ds = CPConjDataset(args.root, args.mask_root, args.img_size, tr_items, augment=True)
    val_ds   = CPConjDataset(args.root, args.mask_root, args.img_size, va_items, augment=False)

    pin_mem = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=Cfg.num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=Cfg.num_workers, pin_memory=pin_mem)

    model = UNet(in_ch=3, out_ch=1, base=32).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(Cfg.amp and device.startswith("cuda")))

    Cfg.out_dir.mkdir(parents=True, exist_ok=True)
    Cfg.pred_dir.mkdir(parents=True, exist_ok=True)

    best_val = 1e9; best_iou = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        tl, tiou = 0.0, 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(Cfg.amp and device.startswith("cuda"))):
                logits = model(imgs)
                loss = bce(logits, masks) + dice(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += loss.item()*imgs.size(0)
            tiou += iou_score(logits.detach(), masks.detach())*imgs.size(0)

        tl /= len(train_ds); tiou /= len(train_ds)

        # validation
        model.eval()
        vl, viou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                vloss = bce(logits, masks) + dice(logits, masks)
                vl += vloss.item()*imgs.size(0)
                viou += iou_score(logits, masks)*imgs.size(0)
        vl /= len(val_ds); viou /= len(val_ds)

        print(f"[{epoch:03d}/{args.epochs}] train_loss={tl:.4f} val_loss={vl:.4f}  train_iou={tiou:.3f} val_iou={viou:.3f}")

        if vl < best_val:
            best_val = vl
            cfg_dump = {k: v for k, v in vars(Cfg).items() if not k.startswith("__")}
            torch.save({"model": model.state_dict(), "cfg": cfg_dump}, Cfg.out_dir/"unet_cp_best.pt")
        if viou > best_iou:
            best_iou = viou
            cfg_dump = {k: v for k, v in vars(Cfg).items() if not k.startswith("__")}
            torch.save({"model": model.state_dict(), "cfg": cfg_dump}, Cfg.out_dir/"unet_cp_best_iou.pt")

        if epoch % Cfg.save_every_pred == 0:
            save_preds(model, val_loader, device, Cfg.pred_dir/("ep_%03d"%epoch), max_save=6)

    print(f"Done. best_val={best_val:.4f}, best_iou={best_iou:.3f}, saved to {Cfg.out_dir}")

if __name__ == "__main__":
    args = parse_args()
    Cfg.device = str(args.device)
    train(args)
