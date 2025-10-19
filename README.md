# Training Scripts

This repo contains various ML training scripts I've collected and built over time. Everything was scattered around, so I organized it into folders to make it easier to find what you need.

## What's Here

I've got training scripts for different approaches:

- **YOLO12** - Latest YOLO with area attention, good for most detection tasks
- **RT-DETR** - Transformer-based detector, great for real-time stuff
- **Faster R-CNN** - Classic two-stage detector, high accuracy but slower
- **Deep SVDD** - One-class classification for anomaly detection

Plus some tools for data augmentation and cloud training on RunPod.

## Folder Structure

```
training/
├── yolo12/          # YOLO12 scripts
├── rtdetr/          # RT-DETR scripts  
├── fasterrcnn/      # Faster R-CNN scripts
└── deep_svdd/       # One-class classification

tools/
├── gui/             # GUI applications
└── dataset_utils/   # Dataset processing

cloud/runpod/        # RunPod cloud training
requirements/        # All the pip requirements
configs/             # Config files
examples/            # Example scripts
```

## Getting Started

Pick what you want to train and go to that folder. Each has its own README with specific instructions.

### YOLO12 (probably what you want)
```bash
cd training/yolo12
pip install -r ../../requirements/requirements_yolo12.txt
python yolo12_train.py --model yolo12n --data coco8.yaml --epochs 100
```

### RT-DETR (if you need real-time + accuracy)
```bash
cd training/rtdetr
pip install -r ../../requirements/requirements_rtdetr.txt
python rtdetr_train.py --model rtdetr-l --data coco8.yaml --epochs 100
```

### Faster R-CNN (if you need maximum accuracy)
```bash
cd training/fasterrcnn
pip install -r ../../requirements/requirements.txt
python train_fasterrcnn.py --data_dir /path/to/dataset --output_dir ./outputs
```

### Deep SVDD (for anomaly detection)
```bash
cd training/deep_svdd
pip install -r ../../requirements/requirements_deep_svdd.txt
python deep_svdd_gui.py  # Has a GUI
```

## Cloud Training

If you want to train on RunPod:
```bash
cd cloud/runpod
pip install -r ../../requirements/requirements_runpod.txt
python runpod_launcher.py
```

## Tools

- **Image Augmentation GUI**: `tools/gui/image_augmentation_gui.py` - Preview augmentations before applying them
- **Dataset Converter**: `tools/dataset_utils/yolo12_dataset_converter.py` - Convert between dataset formats

## Requirements

Each training approach has its own requirements file in the `requirements/` folder. Install the one you need:

```bash
# Core stuff
pip install -r requirements/requirements.txt

# Or specific to what you're training
pip install -r requirements/requirements_yolo12.txt
```

## CLI Package

There's also a CLI package you can install:
```bash
pip install -e .
rfdetect --help
rfdetect-gui
```

## Notes

- Most scripts support COCO, VOC, and YOLO dataset formats
- GPU recommended but not required (CPU will just be slower)
- Check individual READMEs in each training folder for more details
- The RunPod stuff is set up for cloud training if you don't have a good GPU locally