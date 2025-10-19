# YOLO12 Nails Training Script
# ============================

Write-Host "🚀 Starting YOLO12 training on nails dataset..." -ForegroundColor Green

# Set dataset path
$datasetPath = "C:\Users\samps\OneDrive\Desktop\nails_dataset"

# Check if dataset exists
if (-not (Test-Path $datasetPath)) {
    Write-Host "❌ Dataset not found at: $datasetPath" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Dataset found at: $datasetPath" -ForegroundColor Yellow

# Create dataset YAML first
Write-Host "📝 Creating dataset YAML..." -ForegroundColor Yellow
python yolo12_train.py --create-dataset-yaml $datasetPath

# Check if YAML was created
if (Test-Path "yolo12_dataset.yaml") {
    Write-Host "✅ Dataset YAML created successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create dataset YAML" -ForegroundColor Red
    exit 1
}

# Start training
Write-Host "🏃 Starting YOLO12 training..." -ForegroundColor Green
Write-Host "Model: YOLO12s (balanced performance)" -ForegroundColor Cyan
Write-Host "Epochs: 200" -ForegroundColor Cyan
Write-Host "Batch size: 24" -ForegroundColor Cyan
Write-Host "Image size: 960" -ForegroundColor Cyan

python yolo12_train.py --model yolo12s --data yolo12_dataset.yaml --epochs 200 --batch 24 --imgsz 960

Write-Host "✅ Training completed!" -ForegroundColor Green
Write-Host "📁 Results saved to: runs/train/yolo12" -ForegroundColor Yellow






