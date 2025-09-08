import os
from glob import glob
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

from datasets import ConjAnemiaDataset
from unet import UNet
from roi_utils import detect_eye_region


def _make_loaders(img_root, mask_root, mask_suffix, img_size, batch, augment):
    ds = ConjAnemiaDataset(img_root, mask_root, mask_suffix,
                           img_size=img_size, augment=augment)
    val_len = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_len, val_len],
        generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)
    return train_loader, val_loader


def _epoch(model, loader, opt=None, device="cuda"):
    if opt:
        model.train()
    else:
        model.eval()
    tot_loss, tot_iou, count = 0, 0, 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = F.binary_cross_entropy(pred, mask)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        iou = ((pred > 0.5) & (mask > 0.5)).float().sum() / \
              ((pred > 0.5) | (mask > 0.5)).float().sum().clamp_min(1.0)
        tot_loss += loss.item() * len(img)
        tot_iou += iou.item() * len(img)
        count += len(img)
    return tot_loss / count, tot_iou / count


def train_stage1(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    train_loader, val_loader = _make_loaders(args.img_root, args.mask_root,
                                             args.mask_suffix, args.img_size,
                                             args.batch, augment=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    best_iou, best_w = 0.0, args.out
    for ep in range(1, args.epochs + 1):
        tl, tiou = _epoch(model, train_loader, opt, device)
        vl, viou = _epoch(model, val_loader, None, device)
        print(f"[Stage1][Ep {ep:03d}] Train {tl:.3f}/{tiou:.3f}  Val {vl:.3f}/{viou:.3f}")
        if viou > best_iou:
            best_iou = viou
            torch.save(model.state_dict(), best_w)
    print("Stage-1 best model saved:", best_w)


def finetune_stage2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    train_loader, val_loader = _make_loaders(args.img_root, args.mask_root,
                                             args.mask_suffix, args.img_size,
                                             args.batch, augment=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    best_iou, best_w = 0.0, args.out
    for ep in range(1, args.epochs + 1):
        tl, tiou = _epoch(model, train_loader, opt, device)
        vl, viou = _epoch(model, val_loader, None, device)
        print(f"[Stage2][Ep {ep:03d}] Train {tl:.3f}/{tiou:.3f}  Val {vl:.3f}/{viou:.3f}")
        if viou > best_iou:
            best_iou = viou
            torch.save(model.state_dict(), best_w)
    print("Stage-2 best model saved:", best_w)


def run_inference(args):
    from pathlib import Path

    img_root = Path(args.img_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    # 모든 하위 폴더까지 재귀적으로 이미지 탐색
    img_files = [p for p in img_root.rglob("*") if p.suffix.lower() in exts]

    iou_scores, dice_scores = [], []

    for fp in tqdm(img_files, desc="infer"):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        if args.use_eye_detector:
            x1, y1, x2, y2 = detect_eye_region(img)
            img_crop = img[y1:y2, x1:x2].copy()
        else:
            img_crop = img.copy()

        orig = img_crop.copy()
        im_res = cv2.resize(img_crop, (args.img_size, args.img_size))
        tensor = cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        tensor = torch.tensor(tensor / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor)[0, 0].cpu().numpy()

        mask = (pred > 0.5).astype(np.uint8) * 255
        roi_color = cv2.bitwise_and(im_res, im_res, mask=mask)

        # 원본 폴더 구조 보존
        rel_path = fp.relative_to(img_root).with_suffix("")
        save_dir = out_root / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = rel_path.name
        cv2.imwrite(str(save_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(save_dir / f"{stem}_roi.png"), roi_color)

        # --- 정답 마스크가 존재하면 평가 ---
        gt_mask_path = fp.with_name(f"{fp.stem}{args.mask_suffix}.png")
        if gt_mask_path.exists():
            mask_gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gt is not None:
                mask_gt = cv2.resize(mask_gt, (args.img_size, args.img_size))
                mask_pr = cv2.resize(mask, (args.img_size, args.img_size))

                inter = np.logical_and(mask_gt > 127, mask_pr > 127).sum()
                union = np.logical_or(mask_gt > 127, mask_pr > 127).sum()
                iou = inter / union if union > 0 else 0
                dice = (2 * inter) / (mask_gt.sum() + mask_pr.sum() + 1e-6)

                iou_scores.append(iou)
                dice_scores.append(dice)

    # --- 최종 성능 출력 ---
    if iou_scores and dice_scores:
        print(f"Mean IoU: {np.mean(iou_scores):.4f}")
        print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    else:
        print("⚠️ No ground truth masks were found for evaluation.")
