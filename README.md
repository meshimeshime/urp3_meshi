# urp3_meshi

U-Net training and inference scripts for the **Conjunctival Images for Anemia Detection** dataset.

## Files
- `train.py`: command-line tool to train a stage-1 U-Net for conjunctival ROI segmentation.
- `inference_roi.py`: generate ROI masks and bounding-box JSON from a trained model.
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
python inference_roi.py \
    --img-root "Conjunctival Images for Anemia Detection" \
    --ckpt stage1.pt \
    --out-dir preds
```

For each input image the script saves:
- `*_mask.png`: binary ROI mask
- `*_roi.png`: color ROI image
- `*_bbox.json`: bounding box and centroid point for SlimSAM

