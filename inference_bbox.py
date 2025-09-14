import os
import json
import torch
import cv2
import numpy as np
from glob import glob

from unet import UNet


def _load_img(path, img_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    tensor = torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0)
    return tensor, img


def run_inference_bbox(img_root, ckpt, out_dir="preds_bbox", img_size=256):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_c=3, out_c=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    img_files = [p for p in glob(os.path.join(img_root, "**", "*.jpg"), recursive=True)]
    print(f"[DEBUG] Found {len(img_files)} jpg images for ROI inference")

    for fp in img_files:
        tensor, orig = _load_img(fp, img_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            pred = model(tensor)[0, 0].cpu().numpy()

        mask = (pred > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + w - 1, y + h - 1
        else:
            x1, y1, x2, y2 = 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bbox_data = {"bbox": [int(x1), int(y1), int(x2), int(y2)], "point": [int(cx), int(cy)]}

        rel = os.path.relpath(fp, img_root)
        base = os.path.splitext(rel)[0]
        out_bbox = os.path.join(out_dir, f"{base}_bbox.json")
        os.makedirs(os.path.dirname(out_bbox), exist_ok=True)

        with open(out_bbox, "w") as f:
            json.dump(bbox_data, f)

        print(f"Saved: {out_bbox}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict conjunctival ROI and export bounding boxes")
    parser.add_argument("--img-root", required=True, help="Root directory containing full-eye images")
    parser.add_argument("--ckpt", required=True, help="Path to trained U-Net model")
    parser.add_argument("--out-dir", default="preds_bbox", help="Directory to save bbox JSON files")
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    run_inference_bbox(args.img_root, args.ckpt, args.out_dir, args.img_size)
