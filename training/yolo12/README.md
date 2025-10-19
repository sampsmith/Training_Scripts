# YOLO12 Training

YOLO12 is the newest YOLO version with some improvements like area attention and R-ELAN architecture. It's probably what you want for most object detection tasks.

## Quick Start

```bash
# Install the requirements
pip install -r ../../requirements/requirements_yolo12.txt

# Train a small model on COCO
python yolo12_train.py --model yolo12n --data coco8.yaml --epochs 100

# Or use your own dataset
python yolo12_train.py --model yolo12s --data /path/to/your/dataset.yaml --epochs 200
```

## Model Sizes

- `yolo12n` - Nano (fastest, smallest)
- `yolo12s` - Small (good balance)
- `yolo12m` - Medium (better accuracy)
- `yolo12l` - Large (high accuracy)
- `yolo12x` - Extra Large (best accuracy, slowest)

## What's Different

- Area attention mechanism for better feature extraction
- R-ELAN architecture improvements
- Optional FlashAttention for faster training
- Works with COCO, VOC, and YOLO dataset formats
- Better logging and visualization

## Files

- `yolo12_train.py` - Main training script with all options
- `yolo12_quick_start.py` - Simpler version for quick tests
- `yolo12_dataset_converter.py` - Convert between dataset formats

Check `../../configs/yolo12_dataset.yaml` for an example config file.
