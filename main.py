import argparse
from train import train_stage1, finetune_stage2, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Conjunctiva ROI Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    s1 = sub.add_parser("train-stage1")
    s1.add_argument("--img-root", required=True)
    s1.add_argument("--mask-root", required=True)
    s1.add_argument("--mask-suffix", default="", help="mask file suffix, e.g. _palpebral")
    s1.add_argument("--img-size", type=int, default=256)
    s1.add_argument("--epochs", type=int, default=40)
    s1.add_argument("--batch", type=int, default=8)
    s1.add_argument("--lr", type=float, default=1e-3)
    s1.add_argument("--out", default="stage1.pt")

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

    inf = sub.add_parser("infer")
    inf.add_argument("--img-root", required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--img-size", type=int, default=256)
    inf.add_argument("--out-dir", default="preds")
    inf.add_argument("--use-eye-detector", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train-stage1":
        train_stage1(args)
    elif args.cmd == "finetune-stage2":
        finetune_stage2(args)
    elif args.cmd == "infer":
        run_inference(args)
    else:
        print("사용법: train-stage1 | finetune-stage2 | infer")


if __name__ == "__main__":
    main()
