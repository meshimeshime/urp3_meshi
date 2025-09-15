import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from unet import UNet
from datasets import ImageFolderDataset


# -----------------------------
# Stage1: ROI Segmentation 학습
# -----------------------------
def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderDataset(
        args.img_root, args.mask_root, args.mask_suffix, img_size=args.img_size
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = UNet(in_c=3, out_c=3).to(device)  # RGB 출력
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Stage1][Ep {epoch+1:03d}] Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.out)
            print(f"Stage-1 best model saved: {args.out}")


# -----------------------------
# Stage2: ROI 기준 Segmentation Fine-tune
# -----------------------------
def finetune_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderDataset(
        args.img_root, args.mask_root, args.mask_suffix, img_size=args.img_size
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = UNet(in_c=3, out_c=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Stage2][Ep {epoch+1:03d}] Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.out)
            print(f"Stage-2 best model saved: {args.out}")


# -----------------------------
# Stage2: Autoencoder Fine-tune (CP-AnemiC 기준)
# -----------------------------
class AutoencoderDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.img_files = []
        for subdir, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.img_files.append(os.path.join(subdir, f))
        self.img_size = img_size
        print(f"[DEBUG] Found {len(self.img_files)} images under {root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, img  # 입력 = 타겟


def finetune_stage2_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AutoencoderDataset(args.img_root, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = UNet(in_c=3, out_c=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)

            recons = model(imgs)
            loss = criterion(recons, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Stage2-AE][Ep {epoch+1:03d}] Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.out)
            print(f"Stage-2 AE best model saved: {args.out}")


# -----------------------------
# Inference (IoU / Dice 계산 포함)
# -----------------------------
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderDataset(
        args.img_root, args.mask_root, args.mask_suffix, img_size=args.img_size
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # checkpoint랑 동일하게 out_c=3
    model = UNet(in_c=3, out_c=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    iou_scores, dice_scores = [], []
    for i, (img, mask_gt) in enumerate(loader):
        img, mask_gt = img.to(device), mask_gt.to(device)
        with torch.no_grad():
            pred = model(img)   # [B,3,H,W]
            pred = torch.sigmoid(pred[:,0:1,:,:])  # 첫 채널만 사용

        pred_bin = (pred > 0.5).float()

        stem = f"{i:03d}"
        out_path = os.path.join(args.out_dir, f"{stem}_roi.png")
        out_img = (pred_bin[0,0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(out_path, out_img)

        inter = (pred_bin * mask_gt).sum().item()
        union = ((pred_bin + mask_gt) > 0).sum().item()
        iou = inter / union if union > 0 else 0
        dice = (2 * inter) / (pred_bin.sum().item() + mask_gt.sum().item() + 1e-6)

        iou_scores.append(iou)
        dice_scores.append(dice)

    if iou_scores and dice_scores:
        print(f"Mean IoU: {np.mean(iou_scores):.4f}")
        print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    else:
        print("⚠️ No ground truth masks were found for evaluation.")

