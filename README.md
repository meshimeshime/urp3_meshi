# urp3_meshi

U-Net training and inference scripts for the **Conjunctival Images for Anemia Detection** dataset.

## Files
- `train.py`: command-line tool to train a stage-1 U-Net for conjunctival ROI segmentation.
- `inference_bbox.py`: predict conjunctival ROI and export bounding-box JSON.
- `datasets.py`: simple dataset loader for full-eye images and `_palpebral.png` masks.
- `unet.py`: lightweight U-Net model.

## Usage
### Train U-Net (stage-1)
```
python train.py \
    --img-root "Conjunctival Images for Anemia Detection" \
    --mask-root "Conjunctival Images for Anemia Detection" \
    --mask-suffix _palpebral \
    --out stage1.pt
```

### Inference and bounding-box extraction
```
python inference_bbox.py \
    --img-root "Conjunctival Images for Anemia Detection" \
    --ckpt stage1.pt \
    --out-dir preds_bbox
```

The script stores `<image>_bbox.json` files under the output directory, mirroring
the input folder structure. Each JSON contains the bounding box `[x1, y1, x2, y2]`
and its center point `[cx, cy]` suitable for SlimSAM.

