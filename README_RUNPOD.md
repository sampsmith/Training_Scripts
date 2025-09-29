# RunPod Object Detection Training

This repository contains optimized training scripts for object detection on RunPod cloud infrastructure.

## 🚀 Quick Start

### 1. Setup (One-time)
```bash
# Install dependencies
pip install -r requirements_runpod.txt

# Run setup script
python setup_runpod.py
```

### 2. Configure Paths
Edit `runpod_train.py` and update these paths:
```python
config = RunPodConfig(
    data_dir="/workspace/dataset",  # Your dataset path
    output_dir="/workspace/outputs",  # Output path
    epochs=50,  # Number of epochs
    patience=10,  # Early stopping patience
)
```

### 3. Start Training
```bash
python runpod_train.py
```

## 📁 Dataset Structure

The script automatically detects COCO or VOC format:

### COCO Format (Roboflow Export)
```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

### VOC Format
```
dataset/
├── train/
│   ├── *.jpg
│   └── *.xml
├── valid/
│   ├── *.jpg
│   └── *.xml
└── test/
    ├── *.jpg
    └── *.xml
```

## ⚙️ Configuration

### Key Settings (Hardcoded for Optimal Performance)

| Setting | Value | Description |
|---------|-------|-------------|
| Model | Faster R-CNN ResNet50-FPN | Best for object detection |
| Pretrained | COCO | Pre-trained on COCO dataset |
| Batch Size | Auto-detected | Automatically finds optimal size |
| Mixed Precision | Enabled | Faster training, less memory |
| Gradient Clipping | 1.0 | Prevents gradient explosion |
| Learning Rate | 0.005 | Optimized for detection |
| Scheduler | StepLR | Reduces LR every 8 epochs |

### Cloud Optimizations

- **Automatic Batch Size Detection**: Finds the largest batch size that fits in GPU memory
- **Memory Management**: Automatic cache clearing and memory optimization
- **Robust Error Handling**: Recovers from OOM errors and other issues
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Checkpoint Management**: Automatic saving and best model tracking

## 📊 Monitoring

### Logs
- Training logs: `/workspace/outputs/logs/training.log`
- Real-time progress in terminal
- Detailed system information logging

### Checkpoints
- Regular checkpoints: `/workspace/outputs/checkpoints/epoch_XXX.pth`
- Best model: `/workspace/outputs/checkpoints/best_model.pth`
- Automatic saving every 5 epochs

### Metrics
- **COCO Format**: Uses COCO AP (Average Precision)
- **VOC Format**: Uses mean confidence score
- **Early Stopping**: Stops if no improvement for 10 epochs

## 🔧 RunPod Specific Features

### Memory Optimization
- Automatic batch size detection
- Mixed precision training (AMP)
- Gradient accumulation for effective large batch sizes
- Memory cleanup after each batch

### Error Recovery
- OOM error handling with automatic batch size reduction
- Checkpoint resuming capability
- Comprehensive error logging

### Cloud Performance
- Optimized data loading with multiple workers
- Pinned memory for faster GPU transfers
- Efficient checkpoint saving

## 📈 Training Process

1. **Dataset Detection**: Automatically detects COCO or VOC format
2. **Model Building**: Creates Faster R-CNN with optimal settings
3. **Batch Size Detection**: Tests different batch sizes to find optimal
4. **Training Loop**: 
   - Mixed precision training
   - Gradient accumulation
   - Learning rate scheduling
   - Regular evaluation
   - Checkpoint saving
5. **Early Stopping**: Stops if no improvement

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- The script automatically handles this by reducing batch size
- Check GPU memory: `nvidia-smi`

**Dataset Not Found**
- Verify dataset path in `runpod_train.py`
- Ensure dataset has correct structure (train/valid folders)

**Training Stops Early**
- Check patience setting (default: 10 epochs)
- Monitor validation metrics in logs

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Increase num_workers** if you have more CPU cores
3. **Monitor GPU utilization** with `nvidia-smi`
4. **Use mixed precision** (enabled by default)

## 📋 Output Files

```
/workspace/outputs/
├── checkpoints/
│   ├── epoch_005.pth
│   ├── epoch_010.pth
│   └── best_model.pth
├── logs/
│   └── training.log
└── runpod_config.json
```

## 🔄 Resuming Training

To resume from a checkpoint:
```python
# In runpod_train.py, add resume path
config = RunPodConfig(
    # ... other settings
    resume_path="/workspace/outputs/checkpoints/epoch_010.pth"
)
```

## 📞 Support

For issues or questions:
1. Check the logs in `/workspace/outputs/logs/`
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Check GPU memory and availability

## 🎯 Best Practices

1. **Dataset Quality**: Ensure high-quality annotations
2. **Data Augmentation**: Script includes horizontal flipping
3. **Regular Monitoring**: Check logs and metrics regularly
4. **Backup Checkpoints**: Save important checkpoints
5. **GPU Monitoring**: Watch GPU utilization and memory usage

---

**Happy Training! 🚀**
