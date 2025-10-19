# RT-DETR Training

RT-DETR is a transformer-based detector that's good for real-time stuff. It uses vision transformers but is optimized for speed.

## Quick Start

```bash
# Install requirements
pip install -r ../../requirements/requirements_rtdetr.txt

# Train the large model
python rtdetr_train.py --model rtdetr-l --data coco8.yaml --epochs 100

# Or use your own dataset
python rtdetr_train.py --model rtdetr-x --data /path/to/your/dataset.yaml --epochs 200
```

## Model Options

- `rtdetr-l` - Large (good balance of speed and accuracy)
- `rtdetr-x` - Extra Large (best accuracy, slower)

## Why Use This

- Good for real-time applications
- Transformer architecture (attention mechanisms)
- No NMS needed (faster inference)
- Works with COCO, VOC, YOLO formats
- Ready for TensorRT acceleration

## Files

- `rtdetr_train.py` - Main training script
- `rtdetr_quick_start.py` - Simpler version for testing

## When to Use

- Need real-time detection with good accuracy
- Want to use transformers for detection
- Planning to deploy with TensorRT
- Need both speed and precision
