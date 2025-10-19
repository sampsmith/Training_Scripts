#!/usr/bin/env python3
"""
Deep SVDD One-Class Classification GUI
======================================

PySide6 application for training Deep SVDD models for one-class classification.
Features:
- Interactive parameter configuration
- Real-time training progress monitoring
- Loss visualisation
- Model saving and loading
- Dataset preview

Usage:
    python deep_svdd_gui.py
"""

import sys
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QFileDialog, QGroupBox, QGridLayout,
    QMessageBox, QTabWidget, QComboBox, QCheckBox, QListWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QImage

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class FlatDirectoryDataset(Dataset):
    """Custom dataset for loading images from a flat directory (no class folders needed)"""
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Directory with all images
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
        
        # Find all images
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
            # Also search in subdirectories
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        self.image_paths = list(set(self.image_paths))  # Remove duplicates
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image fails to load, create a blank image
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and dummy label (0) since this is one-class
        return image, 0


class TrainingWorker(QThread):
    """Worker thread for training to avoid UI freezing"""
    
    # Signals for updating UI
    progress_update = Signal(int, int, float)  # epoch, total_epochs, loss
    training_complete = Signal(str)  # message
    training_error = Signal(str)  # error message
    log_message = Signal(str)  # log text
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.should_stop = False
        
    def run(self):
        """Main training loop running in separate thread"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"Using device: {device}")
            
            # Load dataset
            self.log_message.emit(f"Loading dataset from: {self.config['data_dir']}")
            
            transform = transforms.Compose([
                transforms.Resize((self.config['img_size'], self.config['img_size'])),
                transforms.RandomRotation(self.config['rotation']),
                transforms.ColorJitter(
                    brightness=self.config['brightness'],
                    contrast=self.config['contrast']
                ),
                transforms.ToTensor(),
            ])
            
            try:
                # Try flat directory first (for one-class)
                dataset = FlatDirectoryDataset(root_dir=self.config['data_dir'], transform=transform)
                
                if len(dataset) == 0:
                    # If no images found, try ImageFolder format
                    self.log_message.emit("No images in flat directory, trying ImageFolder format...")
                    dataset = datasets.ImageFolder(root=self.config['data_dir'], transform=transform)
                
                if len(dataset) == 0:
                    self.training_error.emit("No images found in the specified directory!")
                    return
                
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.config['batch_size'], 
                    shuffle=True,
                    num_workers=self.config['num_workers']
                )
                self.log_message.emit(f"Dataset loaded: {len(dataset)} images")
            except Exception as e:
                self.training_error.emit(f"Failed to load dataset: {str(e)}")
                return
            
            # Create model
            self.log_message.emit(f"Creating {self.config['backbone']} backbone...")
            
            if self.config['backbone'] == 'ResNet18':
                backbone = models.resnet18(pretrained=self.config['pretrained'])
                backbone.fc = nn.Linear(backbone.fc.in_features, self.config['embedding_dim'])
            elif self.config['backbone'] == 'ResNet34':
                backbone = models.resnet34(pretrained=self.config['pretrained'])
                backbone.fc = nn.Linear(backbone.fc.in_features, self.config['embedding_dim'])
            elif self.config['backbone'] == 'ResNet50':
                backbone = models.resnet50(pretrained=self.config['pretrained'])
                backbone.fc = nn.Linear(backbone.fc.in_features, self.config['embedding_dim'])
            else:
                backbone = models.resnet18(pretrained=self.config['pretrained'])
                backbone.fc = nn.Linear(backbone.fc.in_features, self.config['embedding_dim'])
            
            backbone = backbone.to(device)
            
            # Initialise hypersphere centre
            c = torch.zeros(self.config['embedding_dim'], device=device)
            
            # Optimiser
            optimizer = optim.Adam(backbone.parameters(), lr=self.config['lr'])
            
            # Training loop
            self.log_message.emit("Starting training...")
            
            for epoch in range(self.config['num_epochs']):
                if self.should_stop:
                    self.log_message.emit("Training stopped by user")
                    break
                    
                backbone.train()
                epoch_loss = 0
                num_batches = 0
                
                for batch_idx, (images, _) in enumerate(dataloader):
                    if self.should_stop:
                        break
                        
                    images = images.to(device)
                    embeddings = backbone(images)
                    
                    # Update centre after first batch
                    if epoch == 0 and batch_idx == 0:
                        c = torch.mean(embeddings.detach(), dim=0)
                        self.log_message.emit("Hypersphere centre initialised")
                    
                    # Deep SVDD loss
                    dist = torch.sum((embeddings - c) ** 2, dim=1)
                    loss = torch.mean(dist)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * images.size(0)
                    num_batches += 1
                
                if self.should_stop:
                    break
                
                epoch_loss /= len(dataset)
                
                # Emit progress update
                self.progress_update.emit(epoch + 1, self.config['num_epochs'], epoch_loss)
                self.log_message.emit(
                    f"Epoch {epoch+1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}"
                )
            
            if not self.should_stop:
                # Save model
                output_path = self.config['output_path']
                torch.save({
                    'model_state_dict': backbone.state_dict(),
                    'center': c,
                    'config': self.config
                }, output_path)
                
                self.training_complete.emit(f"Training complete! Model saved to: {output_path}")
            
        except Exception as e:
            self.training_error.emit(f"Training error: {str(e)}")
    
    def stop(self):
        """Stop training"""
        self.should_stop = True


class LossPlotWidget(QWidget):
    """Widget for plotting loss over time"""
    
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(8, 4), facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='#2d2d2d')
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.losses = []
        self.epochs = []
        
        self.setup_plot()
    
    def setup_plot(self):
        """Initialise plot with dark theme"""
        self.ax.set_xlabel('Epoch', color='#e0e0e0')
        self.ax.set_ylabel('Loss', color='#e0e0e0')
        self.ax.set_title('Training Loss', color='#ffffff', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.2, color='#555555')
        self.ax.tick_params(colors='#e0e0e0', which='both')
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.spines['top'].set_color('#555555')
        self.ax.spines['left'].set_color('#555555')
        self.ax.spines['right'].set_color('#555555')
        self.figure.tight_layout()
    
    def update_plot(self, epoch: int, loss: float):
        """Update plot with new data"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        self.ax.clear()
        self.ax.plot(self.epochs, self.losses, '#0d7377', linewidth=2)
        self.ax.fill_between(self.epochs, self.losses, alpha=0.3, color='#0d7377')
        self.ax.set_xlabel('Epoch', color='#e0e0e0')
        self.ax.set_ylabel('Loss', color='#e0e0e0')
        self.ax.set_title('Training Loss', color='#ffffff', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.2, color='#555555')
        self.ax.tick_params(colors='#e0e0e0', which='both')
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.spines['top'].set_color('#555555')
        self.ax.spines['left'].set_color('#555555')
        self.ax.spines['right'].set_color('#555555')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear plot data"""
        self.losses = []
        self.epochs = []
        self.ax.clear()
        self.setup_plot()
        self.canvas.draw()


class DeepSVDDGUI(QMainWindow):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        self.worker: Optional[TrainingWorker] = None
        
        # Inference state
        self.inference_model = None
        self.inference_center = None
        self.inference_config = None
        self.inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_test_image = None
        self.current_test_image_path = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialise user interface"""
        self.setWindowTitle("Deep SVDD One-Class Classification")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("Deep SVDD One-Class Classification")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; background-color: transparent; padding: 10px;")
        main_layout.addWidget(title)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3a3a3a;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border: 2px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
        """)
        
        # Training tab
        training_tab = self.create_training_tab()
        self.tab_widget.addTab(training_tab, "🎓 Training")
        
        # Inference tab
        inference_tab = self.create_inference_tab()
        self.tab_widget.addTab(inference_tab, "🔍 Inference")
        
        main_layout.addWidget(self.tab_widget)
        
        # Apply stylesheet
        self.apply_stylesheet()
    
    def create_training_tab(self) -> QWidget:
        """Create the training tab"""
        tab = QWidget()
        layout = QHBoxLayout()
        tab.setLayout(layout)
        
        # Left panel - Configuration
        left_panel = self.create_config_panel()
        layout.addWidget(left_panel, stretch=1)
        
        # Right panel - Monitoring
        right_panel = self.create_monitor_panel()
        layout.addWidget(right_panel, stretch=2)
        
        return tab
        
    def create_config_panel(self) -> QWidget:
        """Create configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Configuration")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; background-color: transparent; padding: 10px;")
        layout.addWidget(title)
        
        # Dataset settings
        dataset_group = QGroupBox("Dataset Settings")
        dataset_layout = QGridLayout()
        dataset_group.setLayout(dataset_layout)
        
        # Info label
        info_label = QLabel("Select a folder containing your images\n(No class folders needed for one-class)")
        info_label.setStyleSheet("color: #0d7377; font-size: 10px; font-style: italic;")
        dataset_layout.addWidget(info_label, 0, 0, 1, 3)
        
        # Data directory
        dataset_layout.addWidget(QLabel("Data Directory:"), 1, 0)
        self.data_dir_input = QLineEdit("./nail_dataset")
        dataset_layout.addWidget(self.data_dir_input, 1, 1)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_data_dir)
        dataset_layout.addWidget(self.browse_btn, 1, 2)
        
        # Image count display
        self.image_count_label = QLabel("Images found: 0")
        self.image_count_label.setStyleSheet("color: #0d7377; font-weight: bold;")
        dataset_layout.addWidget(self.image_count_label, 2, 0, 1, 3)
        
        # Check images button
        self.check_images_btn = QPushButton("Check Images in Directory")
        self.check_images_btn.clicked.connect(self.check_image_count)
        self.check_images_btn.setStyleSheet("background-color: #3a3a3a; padding: 5px;")
        dataset_layout.addWidget(self.check_images_btn, 3, 0, 1, 3)
        
        # Image size
        dataset_layout.addWidget(QLabel("Image Size:"), 4, 0)
        self.img_size_input = QSpinBox()
        self.img_size_input.setRange(64, 512)
        self.img_size_input.setValue(128)
        dataset_layout.addWidget(self.img_size_input, 4, 1, 1, 2)
        
        layout.addWidget(dataset_group)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout()
        model_group.setLayout(model_layout)
        
        # Backbone
        model_layout.addWidget(QLabel("Backbone:"), 0, 0)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(['ResNet18', 'ResNet34', 'ResNet50'])
        model_layout.addWidget(self.backbone_combo, 0, 1, 1, 2)
        
        # Embedding dimension
        model_layout.addWidget(QLabel("Embedding Dim:"), 1, 0)
        self.embedding_dim_input = QSpinBox()
        self.embedding_dim_input.setRange(64, 2048)
        self.embedding_dim_input.setValue(512)
        model_layout.addWidget(self.embedding_dim_input, 1, 1, 1, 2)
        
        # Pretrained
        self.pretrained_check = QCheckBox("Use Pretrained Weights")
        self.pretrained_check.setChecked(True)
        model_layout.addWidget(self.pretrained_check, 2, 0, 1, 3)
        
        layout.addWidget(model_group)
        
        # Training settings
        training_group = QGroupBox("Training Settings")
        training_layout = QGridLayout()
        training_group.setLayout(training_layout)
        
        # Batch size
        training_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 256)
        self.batch_size_input.setValue(16)
        training_layout.addWidget(self.batch_size_input, 0, 1, 1, 2)
        
        # Epochs
        training_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(20)
        training_layout.addWidget(self.epochs_input, 1, 1, 1, 2)
        
        # Learning rate
        training_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setValue(0.0001)
        self.lr_input.setDecimals(5)
        self.lr_input.setSingleStep(0.0001)
        training_layout.addWidget(self.lr_input, 2, 1, 1, 2)
        
        # Workers
        training_layout.addWidget(QLabel("Num Workers:"), 3, 0)
        self.workers_input = QSpinBox()
        self.workers_input.setRange(0, 16)
        self.workers_input.setValue(4)
        training_layout.addWidget(self.workers_input, 3, 1, 1, 2)
        
        layout.addWidget(training_group)
        
        # Augmentation settings
        aug_group = QGroupBox("Augmentation Settings")
        aug_layout = QGridLayout()
        aug_group.setLayout(aug_layout)
        
        # Rotation
        aug_layout.addWidget(QLabel("Rotation (degrees):"), 0, 0)
        self.rotation_input = QSpinBox()
        self.rotation_input.setRange(0, 180)
        self.rotation_input.setValue(30)
        aug_layout.addWidget(self.rotation_input, 0, 1, 1, 2)
        
        # Brightness
        aug_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.brightness_input = QDoubleSpinBox()
        self.brightness_input.setRange(0, 1)
        self.brightness_input.setValue(0.3)
        self.brightness_input.setSingleStep(0.1)
        aug_layout.addWidget(self.brightness_input, 1, 1, 1, 2)
        
        # Contrast
        aug_layout.addWidget(QLabel("Contrast:"), 2, 0)
        self.contrast_input = QDoubleSpinBox()
        self.contrast_input.setRange(0, 1)
        self.contrast_input.setValue(0.3)
        self.contrast_input.setSingleStep(0.1)
        aug_layout.addWidget(self.contrast_input, 2, 1, 1, 2)
        
        layout.addWidget(aug_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()
        output_group.setLayout(output_layout)
        
        output_layout.addWidget(QLabel("Output Path:"), 0, 0)
        self.output_path_input = QLineEdit("deep_svdd_nail.pth")
        output_layout.addWidget(self.output_path_input, 0, 1)
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.browse_output_btn, 0, 2)
        
        layout.addWidget(output_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        return panel
    
    def create_monitor_panel(self) -> QWidget:
        """Create monitoring panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Training Monitor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; background-color: transparent; padding: 10px;")
        layout.addWidget(title)
        
        # System info
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        
        info_label = QLabel(
            f"PyTorch: {torch.__version__}\n"
            f"CUDA Available: {cuda_available}\n"
            f"Device: {device_name}"
        )
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_label = QLabel("Ready to start training")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.loss_label = QLabel("Loss: N/A")
        self.loss_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.loss_label.setStyleSheet("color: #0d7377; background-color: transparent; padding: 5px;")
        progress_layout.addWidget(self.loss_label)
        
        layout.addWidget(progress_group)
        
        # Loss plot
        self.loss_plot = LossPlotWidget()
        layout.addWidget(self.loss_plot, stretch=2)
        
        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group, stretch=1)
        
        return panel
    
    def create_inference_tab(self) -> QWidget:
        """Create the inference tab"""
        tab = QWidget()
        layout = QHBoxLayout()
        tab.setLayout(layout)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Model loading
        model_group = QGroupBox("Model Loading")
        model_layout = QGridLayout()
        model_group.setLayout(model_layout)
        
        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.inf_model_path_input = QLineEdit("deep_svdd_nail.pth")
        model_layout.addWidget(self.inf_model_path_input, 0, 1)
        self.inf_browse_model_btn = QPushButton("Browse")
        self.inf_browse_model_btn.clicked.connect(self.browse_inference_model)
        model_layout.addWidget(self.inf_browse_model_btn, 0, 2)
        
        self.inf_load_model_btn = QPushButton("Load Model")
        self.inf_load_model_btn.clicked.connect(self.load_inference_model)
        self.inf_load_model_btn.setStyleSheet("background-color: #0d7377; padding: 10px; font-weight: bold;")
        model_layout.addWidget(self.inf_load_model_btn, 1, 0, 1, 3)
        
        self.inf_model_status = QLabel("No model loaded")
        self.inf_model_status.setStyleSheet("color: #f39c12; font-weight: bold;")
        model_layout.addWidget(self.inf_model_status, 2, 0, 1, 3)
        
        left_layout.addWidget(model_group)
        
        # Image selection
        image_group = QGroupBox("Test Image")
        image_layout = QVBoxLayout()
        image_group.setLayout(image_layout)
        
        btn_layout = QHBoxLayout()
        self.inf_load_image_btn = QPushButton("Load Image")
        self.inf_load_image_btn.clicked.connect(self.load_test_image)
        btn_layout.addWidget(self.inf_load_image_btn)
        
        self.inf_load_folder_btn = QPushButton("Load Folder")
        self.inf_load_folder_btn.clicked.connect(self.load_test_folder)
        btn_layout.addWidget(self.inf_load_folder_btn)
        image_layout.addLayout(btn_layout)
        
        self.inf_image_list = QListWidget()
        self.inf_image_list.setMaximumHeight(150)
        self.inf_image_list.currentRowChanged.connect(self.on_test_image_selected)
        image_layout.addWidget(self.inf_image_list)
        
        left_layout.addWidget(image_group)
        
        # Threshold settings
        threshold_group = QGroupBox("Detection Threshold")
        threshold_layout = QGridLayout()
        threshold_group.setLayout(threshold_layout)
        
        threshold_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.inf_threshold_input = QDoubleSpinBox()
        self.inf_threshold_input.setRange(0.0, 1000.0)
        self.inf_threshold_input.setValue(10.0)
        self.inf_threshold_input.setDecimals(2)
        self.inf_threshold_input.setSingleStep(0.5)
        self.inf_threshold_input.valueChanged.connect(self.update_inference_classification)
        threshold_layout.addWidget(self.inf_threshold_input, 0, 1)
        
        threshold_info = QLabel("Lower = stricter\nHigher = more lenient")
        threshold_info.setStyleSheet("color: #888888; font-size: 10px; font-style: italic;")
        threshold_layout.addWidget(threshold_info, 1, 0, 1, 2)
        
        left_layout.addWidget(threshold_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        self.inf_distance_label = QLabel("Distance: N/A")
        self.inf_distance_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.inf_distance_label.setStyleSheet("color: #e0e0e0; padding: 10px;")
        self.inf_distance_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.inf_distance_label)
        
        self.inf_classification_label = QLabel("Classification: N/A")
        self.inf_classification_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.inf_classification_label.setAlignment(Qt.AlignCenter)
        self.inf_classification_label.setStyleSheet("padding: 15px; border-radius: 5px;")
        results_layout.addWidget(self.inf_classification_label)
        
        left_layout.addWidget(results_group)
        
        # Batch inference
        batch_group = QGroupBox("Batch Inference")
        batch_layout = QVBoxLayout()
        batch_group.setLayout(batch_layout)
        
        self.inf_batch_btn = QPushButton("Run Batch Inference on All Images")
        self.inf_batch_btn.clicked.connect(self.run_batch_inference)
        self.inf_batch_btn.setStyleSheet("background-color: #8e44ad; padding: 10px;")
        batch_layout.addWidget(self.inf_batch_btn)
        
        self.inf_batch_results = QTextEdit()
        self.inf_batch_results.setReadOnly(True)
        self.inf_batch_results.setMaximumHeight(150)
        batch_layout.addWidget(self.inf_batch_results)
        
        left_layout.addWidget(batch_group)
        
        left_layout.addStretch()
        layout.addWidget(left_panel, stretch=1)
        
        # Right panel - Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        self.inf_image_display = QLabel()
        self.inf_image_display.setAlignment(Qt.AlignCenter)
        self.inf_image_display.setMinimumSize(600, 600)
        self.inf_image_display.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                color: #e0e0e0;
            }
        """)
        self.inf_image_display.setText("Load a model and image to begin testing")
        right_layout.addWidget(self.inf_image_display)
        
        layout.addWidget(right_panel, stretch=2)
        
        return tab
    
    def apply_stylesheet(self):
        """Apply custom dark theme stylesheet"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #0d7377;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #0d7377;
            }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox:focus {
                border: 1px solid #0d7377;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #555555;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: #ffffff;
                selection-background-color: #0d7377;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555555;
                border-radius: 3px;
                background-color: #3a3a3a;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
                border-color: #0d7377;
            }
            QPushButton {
                padding: 8px;
                border-radius: 3px;
                border: none;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QProgressBar {
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                color: #e0e0e0;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #2d2d2d;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background-color: #2d2d2d;
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555555;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #666666;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
    
    def browse_data_dir(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self.data_dir_input.setText(directory)
            self.check_image_count()  # Auto-check images when directory is selected
    
    def check_image_count(self):
        """Check and display the number of images in the selected directory"""
        data_dir = self.data_dir_input.text()
        
        if not os.path.exists(data_dir):
            self.image_count_label.setText("Images found: 0 (directory doesn't exist)")
            self.image_count_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
        
        # Count images using the same logic as FlatDirectoryDataset
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        image_paths = list(set(image_paths))
        count = len(image_paths)
        
        if count == 0:
            self.image_count_label.setText("Images found: 0 (no images in directory)")
            self.image_count_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        else:
            self.image_count_label.setText(f"Images found: {count} ✓")
            self.image_count_label.setStyleSheet("color: #27ae60; font-weight: bold;")
    
    def browse_output_path(self):
        """Browse for output path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Model As", 
            "", 
            "PyTorch Model (*.pth *.pt)"
        )
        if file_path:
            self.output_path_input.setText(file_path)
    
    def start_training(self):
        """Start training process"""
        # Validate inputs
        data_dir = self.data_dir_input.text()
        if not os.path.exists(data_dir):
            QMessageBox.warning(self, "Error", f"Data directory does not exist: {data_dir}")
            return
        
        # Prepare configuration
        config = {
            'data_dir': data_dir,
            'img_size': self.img_size_input.value(),
            'backbone': self.backbone_combo.currentText(),
            'embedding_dim': self.embedding_dim_input.value(),
            'pretrained': self.pretrained_check.isChecked(),
            'batch_size': self.batch_size_input.value(),
            'num_epochs': self.epochs_input.value(),
            'lr': self.lr_input.value(),
            'num_workers': self.workers_input.value(),
            'rotation': self.rotation_input.value(),
            'brightness': self.brightness_input.value(),
            'contrast': self.contrast_input.value(),
            'output_path': self.output_path_input.text(),
        }
        
        # Clear previous results
        self.log_text.clear()
        self.loss_plot.clear_plot()
        self.progress_bar.setValue(0)
        
        # Disable start button, enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Create and start worker thread
        self.worker = TrainingWorker(config)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.training_complete.connect(self.on_training_complete)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.log_message.connect(self.on_log_message)
        self.worker.start()
        
        self.log_text.append("=== Training Started ===")
    
    def stop_training(self):
        """Stop training process"""
        if self.worker:
            self.worker.stop()
            self.log_text.append("Stopping training...")
            self.stop_btn.setEnabled(False)
    
    def on_progress_update(self, epoch: int, total_epochs: int, loss: float):
        """Handle progress update from worker"""
        progress = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Epoch {epoch}/{total_epochs}")
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.loss_plot.update_plot(epoch, loss)
    
    def on_training_complete(self, message: str):
        """Handle training completion"""
        self.log_text.append(f"\n=== {message} ===")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.information(self, "Success", message)
    
    def on_training_error(self, message: str):
        """Handle training error"""
        self.log_text.append(f"\nERROR: {message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", message)
    
    def on_log_message(self, message: str):
        """Handle log message from worker"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    # Inference methods
    def browse_inference_model(self):
        """Browse for trained model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trained Model",
            "",
            "PyTorch Model (*.pth *.pt)"
        )
        if file_path:
            self.inf_model_path_input.setText(file_path)
    
    def load_inference_model(self):
        """Load trained Deep SVDD model"""
        model_path = self.inf_model_path_input.text()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", f"Model file not found: {model_path}")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.inference_device)
            
            # Get config
            self.inference_config = checkpoint['config']
            self.inference_center = checkpoint['center'].to(self.inference_device)
            
            # Recreate model
            backbone_name = self.inference_config['backbone']
            embedding_dim = self.inference_config['embedding_dim']
            
            if backbone_name == 'ResNet18':
                self.inference_model = models.resnet18(pretrained=False)
                self.inference_model.fc = nn.Linear(self.inference_model.fc.in_features, embedding_dim)
            elif backbone_name == 'ResNet34':
                self.inference_model = models.resnet34(pretrained=False)
                self.inference_model.fc = nn.Linear(self.inference_model.fc.in_features, embedding_dim)
            elif backbone_name == 'ResNet50':
                self.inference_model = models.resnet50(pretrained=False)
                self.inference_model.fc = nn.Linear(self.inference_model.fc.in_features, embedding_dim)
            else:
                self.inference_model = models.resnet18(pretrained=False)
                self.inference_model.fc = nn.Linear(self.inference_model.fc.in_features, embedding_dim)
            
            # Load weights
            self.inference_model.load_state_dict(checkpoint['model_state_dict'])
            self.inference_model.to(self.inference_device)
            self.inference_model.eval()
            
            self.inf_model_status.setText(f"Model loaded: {backbone_name}")
            self.inf_model_status.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            QMessageBox.information(
                self,
                "Success",
                f"Model loaded successfully!\n\nBackbone: {backbone_name}\nEmbedding dim: {embedding_dim}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.inf_model_status.setText("Failed to load model")
            self.inf_model_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
    
    def load_test_image(self):
        """Load a single test image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.test_image_list = [file_path]
            self.inf_image_list.clear()
            self.inf_image_list.addItem(os.path.basename(file_path))
            self.inf_image_list.setCurrentRow(0)
            self.process_test_image(file_path)
    
    def load_test_folder(self):
        """Load all images from a folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Test Folder")
        
        if folder_path:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            self.test_image_list = []
            
            for ext in extensions:
                self.test_image_list.extend(glob.glob(os.path.join(folder_path, ext)))
                self.test_image_list.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            self.test_image_list.sort()
            
            if self.test_image_list:
                self.inf_image_list.clear()
                for img_path in self.test_image_list:
                    self.inf_image_list.addItem(os.path.basename(img_path))
                self.inf_image_list.setCurrentRow(0)
                self.process_test_image(self.test_image_list[0])
            else:
                QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
    
    def on_test_image_selected(self, index: int):
        """Handle test image selection"""
        if hasattr(self, 'test_image_list') and 0 <= index < len(self.test_image_list):
            self.process_test_image(self.test_image_list[index])
    
    def process_test_image(self, image_path: str):
        """Process a test image and compute distance"""
        if self.inference_model is None:
            QMessageBox.warning(self, "No Model", "Please load a trained model first.")
            return
        
        try:
            self.current_test_image_path = image_path
            
            # Load and display image
            image = Image.open(image_path).convert('RGB')
            self.current_test_image = image
            
            # Display image
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.inf_image_display.width() - 20,
                self.inf_image_display.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.inf_image_display.setPixmap(scaled_pixmap)
            
            # Preprocess for model
            img_size = self.inference_config['img_size']
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.inference_device)
            
            # Run inference
            with torch.no_grad():
                embedding = self.inference_model(image_tensor)
                distance = torch.sum((embedding - self.inference_center) ** 2).item()
            
            # Display results
            self.inf_distance_label.setText(f"Distance: {distance:.4f}")
            
            # Classify based on threshold
            self.current_distance = distance
            self.update_inference_classification()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")
    
    def update_inference_classification(self):
        """Update classification based on threshold"""
        if not hasattr(self, 'current_distance'):
            return
        
        threshold = self.inf_threshold_input.value()
        distance = self.current_distance
        
        if distance < threshold:
            self.inf_classification_label.setText("✓ NORMAL")
            self.inf_classification_label.setStyleSheet(
                "color: #ffffff; background-color: #27ae60; padding: 15px; "
                "border-radius: 5px; font-size: 18px;"
            )
        else:
            self.inf_classification_label.setText("✗ ANOMALY")
            self.inf_classification_label.setStyleSheet(
                "color: #ffffff; background-color: #e74c3c; padding: 15px; "
                "border-radius: 5px; font-size: 18px;"
            )
    
    def run_batch_inference(self):
        """Run inference on all loaded images"""
        if self.inference_model is None:
            QMessageBox.warning(self, "No Model", "Please load a trained model first.")
            return
        
        if not hasattr(self, 'test_image_list') or not self.test_image_list:
            QMessageBox.warning(self, "No Images", "Please load test images first.")
            return
        
        threshold = self.inf_threshold_input.value()
        img_size = self.inference_config['img_size']
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.inf_batch_results.clear()
        self.inf_batch_results.append("=== Batch Inference Results ===\n")
        
        normal_count = 0
        anomaly_count = 0
        
        try:
            for image_path in self.test_image_list:
                filename = os.path.basename(image_path)
                
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(self.inference_device)
                
                # Run inference
                with torch.no_grad():
                    embedding = self.inference_model(image_tensor)
                    distance = torch.sum((embedding - self.inference_center) ** 2).item()
                
                # Classify
                if distance < threshold:
                    classification = "NORMAL"
                    normal_count += 1
                    color = "green"
                else:
                    classification = "ANOMALY"
                    anomaly_count += 1
                    color = "red"
                
                self.inf_batch_results.append(f"{filename}: {distance:.4f} → {classification}")
            
            self.inf_batch_results.append(f"\n=== Summary ===")
            self.inf_batch_results.append(f"Total images: {len(self.test_image_list)}")
            self.inf_batch_results.append(f"Normal: {normal_count}")
            self.inf_batch_results.append(f"Anomalies: {anomaly_count}")
            self.inf_batch_results.append(f"Threshold: {threshold}")
            
            QMessageBox.information(
                self,
                "Batch Complete",
                f"Processed {len(self.test_image_list)} images\n\n"
                f"Normal: {normal_count}\nAnomalies: {anomaly_count}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch inference failed: {str(e)}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DeepSVDDGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

