# Aorta Segmentation UNet Bundle

## Overview

This MONAI bundle provides a 3D UNet model for 24-class aortic segmentation from CT scans. The model was trained using MONAI framework with comprehensive preprocessing and data augmentation pipeline.

## Model Architecture

- **Network**: 3D UNet
- **Input channels**: 1 (CT grayscale)
- **Output channels**: 24 (background + 23 anatomical regions)
- **Architecture**: channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2

## Preprocessing Pipeline

1. **LoadImaged**: Load NIfTI images
2. **EnsureChannelFirstd**: Ensure channel-first format
3. **ScaleIntensityRanged**: Scale intensity from [-175, 250] HU to [0.0, 1.0]
4. **CropForegroundd**: Crop around foreground region
5. **Orientationd**: Standardize to RAS orientation
6. **Spacingd**: Resample to 1.5×1.5×2.0 mm spacing
7. **EnsureTyped**: Ensure proper tensor types

## Inference

- **Method**: Sliding window inference
- **ROI size**: 96×96×96 voxels
- **Batch size**: 4 patches per batch
- **Overlap**: 50%
- **Post-processing**: Softmax activation followed by argmax for final segmentation

## Usage

### With MONAI Bundle Runner

```python
from monai.bundle import run

# Run inference
run(run_id="inference", config_file="configs/inference.json", bundle_root=".")
```

### With MONAI Label

This bundle is designed to work with MONAI Label for interactive segmentation in 3D Slicer.

## Model Weights

Place the trained model weights file `best_metric_model.pth` in the `models/` directory.

## Requirements

- MONAI >= 1.3.0
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## License

MIT License - see LICENSE file for details.
