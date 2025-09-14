import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train U-Net for conjunctival ROI segmentation")
    parser.add_argument("--img-root", required=True, help="Root directory with full-eye images")
    parser.add_argument("--mask-root", required=True, help="Directory containing ROI masks")
    parser.add_argument("--mask-suffix", default="_palpebral", help="Suffix appended to image stem for mask files")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="stage1.pt", help="Path to save trained model")
    args = parser.parse_args()

    train_stage1(args)
