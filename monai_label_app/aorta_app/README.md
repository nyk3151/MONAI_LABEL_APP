# MONAI Label Aortic Segmentation App

## Overview

This MONAI Label application provides interactive segmentation capabilities for aortic structures using a pre-trained 3D UNet model with 24 output classes. The application is designed to work with 3D Slicer for interactive annotation and refinement of aortic segmentations.

## Features

- **3D UNet Model**: Pre-trained model with 24 output classes for comprehensive aortic segmentation
- **Interactive Segmentation**: Compatible with 3D Slicer for real-time annotation
- **Preprocessing Pipeline**: Automated intensity scaling, spacing normalization, and orientation standardization
- **Sliding Window Inference**: Efficient inference on large volumes using overlapping patches
- **Real-time Feedback**: Interactive refinement capabilities for improved segmentation quality

## Model Specifications

- **Architecture**: 3D UNet
- **Input**: Single-channel CT scans (Hounsfield Units)
- **Output**: 24-class segmentation (background + 23 anatomical regions)
- **Input Size**: 96×96×96 voxels (patches)
- **Preprocessing**: Intensity scaling [-175, 250] HU → [0, 1], spacing normalization to 1.5×1.5×2.0 mm

## Installation

1. Install MONAI Label:

```bash
pip install monailabel
```

2. Install additional requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the MONAI Label Server

```bash
# Navigate to the app directory
cd monai_label_app

# Start the MONAI Label server
monailabel start_server --app aorta_app --studies /path/to/your/studies --conf models aorta_segmentation
```

### Using with 3D Slicer

1. Install 3D Slicer (version 5.0 or later)
2. Install the MONAI Label extension in 3D Slicer
3. Connect to the MONAI Label server (default: http://localhost:8000)
4. Load your CT images and start interactive segmentation

### Configuration Options

- `--conf models aorta_segmentation`: Use the aortic segmentation model
- `--conf spatial_size [96,96,96]`: Set the inference patch size
- `--conf preload true`: Preload the model for faster inference

## File Structure

```
aorta_app/
├── main.py                 # Main application entry point
├── lib/
│   ├── __init__.py
│   ├── configs.py          # Configuration constants
│   └── infers.py           # Inference task implementation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Integration

To use your trained model weights:

1. Place your trained model file (`best_metric_model.pth`) in the `model/` directory
2. Update the model path in `main.py` if necessary
3. Restart the MONAI Label server

## Customization

### Adding New Models

1. Implement a new inference class in `lib/infers.py`
2. Register the model in `main.py` in the `init_infers()` method
3. Add configuration parameters in `lib/configs.py`

### Modifying Preprocessing

Update the `pre_transforms()` method in `lib/infers.py` to modify the preprocessing pipeline.

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model weights file exists in the correct path
2. **Memory errors**: Reduce the `sw_batch_size` in the inference configuration
3. **Connection issues**: Check that the MONAI Label server is running and accessible

### Logging

Enable debug logging for detailed information:

```bash
monailabel start_server --app aorta_app --studies /path/to/studies --conf models aorta_segmentation --log DEBUG
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please refer to the MONAI Label documentation or create an issue in the repository.
