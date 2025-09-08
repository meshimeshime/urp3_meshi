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

def visualize_reconstruction(img_root, ckpt, out_dir="recon_out", img_size=256, num_samples=10):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_c=3, out_c=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    exts = {".jpg", ".jpeg", ".png"}
    img_files = [p for p in glob(os.path.join(img_root, "**", "*"), recursive=True)
                 if os.path.splitext(p)[1].lower() in exts]

    for i, fp in enumerate(img_files[:num_samples]):
        tensor, orig = _load_img(fp, img_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            recon = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

        # 원본과 복원 이미지를 나란히 붙이기
        concat = np.hstack([orig, recon])

        out_path = os.path.join(out_dir, f"recon_{i}.png")
        cv2.imwrite(out_path, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    visualize_reconstruction(
        img_root="CP-AnemiC dataset",
        ckpt="stage2_auto_color.pt",
        out_dir="recon_out",
        img_size=256,
        num_samples=10
    )
