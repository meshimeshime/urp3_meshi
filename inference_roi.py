import os
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

def run_inference_roi(img_root, ckpt, out_dir="preds_stage2", img_size=256):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_c=3, out_c=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # jpg 파일만 가져오기
    img_files = [p for p in glob(os.path.join(img_root, "**", "*.jpg"), recursive=True)]

    print(f"[DEBUG] Found {len(img_files)} jpg images for ROI inference")

    for fp in img_files:
        tensor, orig = _load_img(fp, img_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            pred = model(tensor)[0,0].cpu().numpy()

        # threshold mask
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        # ROI 추출
        roi = cv2.bitwise_and(orig, orig, mask=mask)

        # 저장
        rel = os.path.relpath(fp, img_root)
        base = os.path.splitext(rel)[0]
        out_mask = os.path.join(out_dir, f"{base}_mask.png")
        out_roi  = os.path.join(out_dir, f"{base}_roi.png")
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)

        cv2.imwrite(out_mask, mask)
        cv2.imwrite(out_roi, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_mask}, {out_roi}")

if __name__ == "__main__":
    run_inference_roi(
        img_root="Conjunctival Images for Anemia Detection",
        ckpt="stage2_auto_color.pt",
        out_dir="preds_stage2",
        img_size=256
    )
