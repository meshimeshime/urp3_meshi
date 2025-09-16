import os
import json
import csv
import torch
import cv2
import numpy as np
from glob import glob
from unet import UNet
from datasets import resize_with_padding
from bbox_utils import mask_to_bbox, draw_bbox


def _load_img(path, img_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_with_padding(img, img_size)
    tensor = torch.tensor(img.transpose(2, 0, 1) / 255.0,
                          dtype=torch.float32).unsqueeze(0)
    return tensor, img


def visualize_reconstruction(img_root, ckpt, out_dir="recon_out", img_size=256, num_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    bbox_json_path = os.path.join(out_dir, "bboxes.json")
    bbox_csv_path = os.path.join(out_dir, "bboxes.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_c=3, out_c=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    exts = {".jpg", ".jpeg"}
    img_files = [p for p in glob(os.path.join(img_root, "**", "*"), recursive=True)
                 if os.path.splitext(p)[1].lower() in exts]

    bboxes = {}   # JSON용 dict
    csv_rows = [("filename", "x", "y", "w", "h")]  # CSV 헤더

    for i, fp in enumerate(img_files[:num_samples]):
        tensor, orig = _load_img(fp, img_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            pred = model(tensor)

        pred_mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

        # 바운딩 박스 추출
        bbox = mask_to_bbox(pred_mask)
        if bbox is not None:
            x, y, w, h = bbox
            # JSON/CSV 저장용
            bboxes[os.path.basename(fp)] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            csv_rows.append((os.path.basename(fp), int(x), int(y), int(w), int(h)))
            # 시각화용
            vis_img = draw_bbox(orig.copy(), bbox, color=(0, 255, 0))
        else:
            vis_img = orig.copy()

        concat = np.hstack([vis_img, cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)])
        out_path = os.path.join(out_dir, f"recon_{i}.png")
        cv2.imwrite(out_path, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")

    # JSON 저장
    with open(bbox_json_path, "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)
    print(f"Bounding boxes saved to {bbox_json_path}")

    # CSV 저장
    with open(bbox_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Bounding boxes saved to {bbox_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="recon_out")  # <- 새 옵션 추가
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    visualize_reconstruction(
        img_root=args.img_root,
        ckpt=args.ckpt,
        out_dir=args.out_dir,
        img_size=args.img_size,
        num_samples=args.num_samples
    )

