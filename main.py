import argparse
import json
from train import train_stage1, finetune_stage2, finetune_stage2_autoencoder, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Conjunctiva ROI Pipeline")
    parser.add_argument("--config", help="Path to JSON config file", default=None)
    sub = parser.add_subparsers(dest="cmd")

    # Stage 1: supervised segmentation
    s1 = sub.add_parser("train-stage1")
    s1.add_argument("--img-root", required=True)
    s1.add_argument("--mask-root", required=True)
    s1.add_argument("--mask-suffix", default="", help="mask file suffix, e.g. _palpebral")
    s1.add_argument("--img-size", type=int, default=256)
    s1.add_argument("--epochs", type=int, default=40)
    s1.add_argument("--batch", type=int, default=8)
    s1.add_argument("--lr", type=float, default=1e-3)
    s1.add_argument("--out", default="stage1.pt")

    # Stage 2: supervised fine-tuning (있을 경우)
    s2 = sub.add_parser("finetune-stage2")
    s2.add_argument("--img-root", required=True)
    s2.add_argument("--mask-root", required=True)
    s2.add_argument("--mask-suffix", default="_mask")
    s2.add_argument("--img-size", type=int, default=256)
    s2.add_argument("--epochs", type=int, default=20)
    s2.add_argument("--batch", type=int, default=8)
    s2.add_argument("--lr", type=float, default=1e-4)
    s2.add_argument("--ckpt", required=True, help="Stage-1 checkpoint path")
    s2.add_argument("--out", default="stage2.pt")

    # Stage 2: autoencoder/SSL fine-tuning (CP-AnemiC ROI 710장 기준)
    s2ae = sub.add_parser("finetune-stage2-ae")
    s2ae.add_argument("--img-root", required=True, help="Path to ROI-only dataset (CP-AnemiC)")
    s2ae.add_argument("--img-size", type=int, default=256)
    s2ae.add_argument("--epochs", type=int, default=20)
    s2ae.add_argument("--batch", type=int, default=8)
    s2ae.add_argument("--lr", type=float, default=1e-4)
    s2ae.add_argument("--ckpt", required=True, help="Stage-1 checkpoint path")
    s2ae.add_argument("--out", default="stage2_auto.pt")

    # Inference
    inf = sub.add_parser("infer")
    inf.add_argument("--img-root", required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--img-size", type=int, default=256)
    inf.add_argument("--out-dir", default="preds")
    inf.add_argument("--use-eye-detector", action="store_true")
    inf.add_argument("--mask-suffix", default="_forniceal_palpebral", help="suffix for ground-truth masks")

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k.replace('-', '_'), v)
    return args


def main():
    args = parse_args()
    if args.cmd == "train-stage1":
        train_stage1(args)
    elif args.cmd == "finetune-stage2":
        finetune_stage2(args)
    elif args.cmd == "finetune-stage2-ae":
        finetune_stage2_autoencoder(args)
    elif args.cmd == "infer":
        run_inference(args)
    else:
        print("사용법: train-stage1 | finetune-stage2 | finetune-stage2-ae | infer")


if __name__ == "__main__":
    main()

