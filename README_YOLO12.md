# YOLO12 Training Script

Complete training solution for [YOLO12](https://docs.ultralytics.com/models/yolo12/) - the latest attention-centric object detection model with significant improvements over previous YOLO versions.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_yolo12.txt
```

### 2. Easy Training (Recommended)
```bash
python yolo12_quick_start.py
```
This will guide you through everything!

### 3. Advanced Training
```bash
python yolo12_train.py --model yolo12n --data dataset.yaml --epochs 100
```

## 🎯 YOLO12 Features

Based on the [official YOLO12 documentation](https://docs.ultralytics.com/models/yolo12/), this implementation includes:

### **Key Innovations:**
- **Area Attention Mechanism**: Efficiently processes large receptive fields with reduced computational cost
- **Residual Efficient Layer Aggregation Networks (R-ELAN)**: Enhanced feature aggregation with residual connections
- **FlashAttention Support**: Optional FlashAttention for memory optimization
- **Optimized Architecture**: Streamlined attention mechanism for better efficiency

### **Performance Improvements:**
- **YOLO12n**: +2.1% mAP over YOLOv10n, +1.2% over YOLO11n
- **YOLO12s**: +1.5% mAP improvement with +42% speed increase vs RT-DETR
- **YOLO12l**: +0.4% mAP over YOLO11l
- **YOLO12x**: +0.6% mAP over YOLO11x

## 📊 Model Variants

| Model | Size | mAP@50-95 | Speed (T4) | Params | FLOPs | Best For |
|-------|------|-----------|------------|--------|-------|----------|
| YOLO12n | 640 | 47.2 | 2.15ms | 3.2M | 8.1B | Edge devices, real-time |
| YOLO12s | 640 | 50.1 | 2.89ms | 9.4M | 25.4B | Balanced performance |
| YOLO12m | 640 | 52.0 | 4.86ms | 20.2M | 67.5B | High accuracy |
| YOLO12l | 640 | 53.7 | 6.77ms | 26.4M | 88.9B | Maximum accuracy |
| YOLO12x | 640 | 55.2 | 11.79ms | 59.1M | 199.0B | Research, best results |

## 📁 Dataset Support

### **Supported Formats:**
- **COCO Format**: `_annotations.coco.json` files
- **YOLO Format**: `labels/` directory with `.txt` files
- **VOC Format**: `.xml` annotation files

### **Dataset Structure:**
```
dataset/
├── train/
│   ├── images/
│   │   └── *.jpg
│   ├── labels/          # YOLO format
│   │   └── *.txt
│   └── _annotations.coco.json  # COCO format
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## ⚙️ Usage Examples

### **Basic Training:**
```bash
python yolo12_train.py --model yolo12n --data coco8.yaml --epochs 100
```

### **Advanced Training:**
```bash
python yolo12_train.py \
    --model yolo12s \
    --data dataset.yaml \
    --epochs 200 \
    --batch 24 \
    --imgsz 640 \
    --lr0 0.01 \
    --device 0 \
    --flash-attention
```

### **Create Dataset YAML:**
```bash
python yolo12_train.py --create-dataset-yaml /path/to/dataset
```

## 🔧 Configuration Options

### **Model Settings:**
- `--model`: Model variant (yolo12n, yolo12s, yolo12m, yolo12l, yolo12x)
- `--data`: Path to dataset YAML file
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Image size (default: 640)

### **Training Settings:**
- `--lr0`: Initial learning rate (default: 0.01)
- `--patience`: Early stopping patience (default: 50)
- `--device`: Device (cpu, 0, 1, 2, ...)
- `--project`: Project directory (default: runs/train)
- `--name`: Experiment name (default: yolo12)

### **YOLO12 Specific:**
- `--flash-attention`: Enable FlashAttention optimization
- `--area-attention-regions`: Number of area attention regions (default: 4)

## 📈 Training Features

### **Automatic Optimizations:**
- **Batch Size Detection**: Automatically finds optimal batch size
- **Learning Rate Scheduling**: Adaptive learning rate with warmup
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Mixed Precision**: Automatic mixed precision training
- **Gradient Clipping**: Prevents gradient explosion

### **Monitoring:**
- **Real-time Logging**: Console and file logging
- **Progress Tracking**: Training progress with tqdm
- **Metrics Visualization**: Automatic plotting of results
- **Checkpoint Management**: Regular and best model saving

### **Validation:**
- **Automatic Validation**: Regular validation during training
- **COCO Metrics**: Standard COCO evaluation metrics
- **Confusion Matrix**: Detailed performance analysis
- **Export Support**: Model export to various formats

## 🚀 Advanced Features

### **FlashAttention (Optional):**
```bash
# Install FlashAttention
pip install flash-attn

# Enable in training
python yolo12_train.py --flash-attention
```

**Requirements for FlashAttention:**
- NVIDIA GPU (Turing, Ampere, Ada Lovelace, or Hopper)
- Compatible CUDA version

### **Multi-GPU Training:**
```bash
python yolo12_train.py --device 0,1,2,3
```

### **Model Export:**
```python
from yolo12_train import YOLO12Trainer, YOLO12Config

config = YOLO12Config(model="yolo12n", data="dataset.yaml")
trainer = YOLO12Trainer(config)
trainer.train_model()

# Export to different formats
trainer.export_model("torchscript")  # TorchScript
trainer.export_model("onnx")         # ONNX
trainer.export_model("tflite")       # TensorFlow Lite
```

## 📊 Performance Tips

### **For Best Results:**
1. **Use appropriate model size** for your dataset
2. **Enable FlashAttention** if you have compatible hardware
3. **Tune batch size** based on your GPU memory
4. **Use data augmentation** for better generalization
5. **Monitor training** with validation metrics

### **Hardware Recommendations:**
- **YOLO12n/s**: RTX 3060, RTX 4060, or better
- **YOLO12m/l**: RTX 3070, RTX 4070, or better
- **YOLO12x**: RTX 3080, RTX 4080, or better

## 🔍 Troubleshooting

### **Common Issues:**

**CUDA Out of Memory:**
```bash
# Reduce batch size
python yolo12_train.py --batch 8

# Use smaller model
python yolo12_train.py --model yolo12n
```

**Dataset Not Found:**
```bash
# Create dataset YAML
python yolo12_train.py --create-dataset-yaml /path/to/dataset
```

**FlashAttention Error:**
```bash
# Disable FlashAttention
python yolo12_train.py --model yolo12n  # (without --flash-attention)
```

## 📚 References

- [YOLO12 Official Documentation](https://docs.ultralytics.com/models/yolo12/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLO12 Paper](https://arxiv.org/abs/2502.12524)

## 🎯 Citation

If you use YOLO12 in your research, please cite:

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

---

**Happy Training with YOLO12! 🚀**

