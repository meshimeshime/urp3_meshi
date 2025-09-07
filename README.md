# urp3_meshi

Two-stage conjunctival ROI segmentation utilities.

## Files
- `main.py`: CLI entry point for training and inference.
- `train.py`: training loops for Stage-1 (full eye) and Stage-2 (ROI fine-tuning).
- `datasets.py`: generic dataset loader supporting mask suffix and augmentation.
- `unet.py`: lightweight U-Net model.
- `roi_utils.py`: optional Haar cascade eye detector used during inference.

## Usage
### Stage-1 (full eye -> ROI)
```
python main.py train-stage1 \
    --img-root ConjunctivalImages \
    --mask-root ConjunctivalImages/palpebral \
    --mask-suffix _palpebral \
    --out stage1.pt
```

### Stage-2 fine-tuning (cropped ROI images)
```
python main.py finetune-stage2 \
    --img-root CP_AnemiC \
    --mask-root CP_AnemiC \
    --mask-suffix _mask \
    --ckpt stage1.pt \
    --out stage2.pt
```

### Inference on new full-eye images
```
python main.py infer \
    --img-root new_images \
    --ckpt stage2.pt \
    --use-eye-detector
```

Prediction files saved in `--out-dir` include the binary mask (`*_mask.png`) and the color ROI (`*_roi.png`) preserving original colours.
