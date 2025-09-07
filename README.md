# urp3_meshi

Utilities for training and evaluating a U-Net model on CP-AnemiC ROI images.

## Prediction outputs

During validation, `save_preds` stores three image types:

- `*_pred.png`: binary mask of the predicted region.
- `*_roi.png`: original image masked by the prediction. Use this file for Hb prediction.
- `*_overlay.png`: overlay of the mask for visualization only.

