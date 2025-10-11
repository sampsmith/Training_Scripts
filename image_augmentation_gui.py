#!/usr/bin/env python3
"""
Image Augmentation GUI Application
==================================

PySide6 application for previewing and applying image augmentations to datasets.
Features:
- Real-time augmentation preview
- Multiple augmentation techniques
- Batch processing
- Before/After comparison
- Save augmented images

Usage:
    python image_augmentation_gui.py
"""

import sys
import os
import random
from pathlib import Path
from typing import Optional, List
import json

import numpy as np
from PIL import Image
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QGridLayout, QScrollArea,
    QMessageBox, QCheckBox, QProgressBar, QListWidget, QComboBox,
    QLineEdit, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QPixmap, QImage


class AugmentationWorker(QThread):
    """Worker thread for batch augmentation"""
    
    progress_update = Signal(int, int, str)  # current, total, filename
    augmentation_complete = Signal(int)  # total_images
    augmentation_error = Signal(str)  # error message
    
    def __init__(self, image_paths: List[str], output_dir: str, config: dict, num_augmentations: int):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.config = config
        self.num_augmentations = num_augmentations
        self.should_stop = False
        
    def run(self):
        """Run batch augmentation"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            total_generated = 0
            
            for idx, image_path in enumerate(self.image_paths):
                if self.should_stop:
                    break
                
                filename = os.path.basename(image_path)
                self.progress_update.emit(idx + 1, len(self.image_paths), filename)
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Save original
                if self.config.get('save_original', True):
                    output_path = os.path.join(self.output_dir, f"orig_{filename}")
                    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    total_generated += 1
                
                # Generate augmentations
                for aug_idx in range(self.num_augmentations):
                    if self.should_stop:
                        break
                    
                    augmented = self.apply_augmentations(image.copy(), self.config)
                    
                    # Save augmented image
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(self.output_dir, f"{name}_aug{aug_idx}{ext}")
                    cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                    total_generated += 1
            
            if not self.should_stop:
                self.augmentation_complete.emit(total_generated)
                
        except Exception as e:
            self.augmentation_error.emit(str(e))
    
    def apply_augmentations(self, image: np.ndarray, config: dict) -> np.ndarray:
        """Apply augmentations based on config"""
        
        # Horizontal flip
        if config.get('horizontal_flip', False) and random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Vertical flip
        if config.get('vertical_flip', False) and random.random() > 0.5:
            image = cv2.flip(image, 0)
        
        # Rotation
        if config.get('rotation', 0) > 0:
            angle = random.uniform(-config['rotation'], config['rotation'])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Brightness
        if config.get('brightness', 0) > 0:
            factor = 1.0 + random.uniform(-config['brightness'], config['brightness'])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Contrast
        if config.get('contrast', 0) > 0:
            factor = 1.0 + random.uniform(-config['contrast'], config['contrast'])
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Saturation
        if config.get('saturation', 0) > 0:
            factor = 1.0 + random.uniform(-config['saturation'], config['saturation'])
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Hue
        if config.get('hue', 0) > 0:
            shift = random.uniform(-config['hue'], config['hue'])
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Gaussian blur
        if config.get('blur', 0) > 0:
            kernel_size = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), config['blur'])
        
        # Noise
        if config.get('noise', 0) > 0:
            noise = np.random.normal(0, config['noise'] * 25.5, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Random crop
        if config.get('random_crop', False):
            h, w = image.shape[:2]
            crop_h = int(h * random.uniform(0.8, 1.0))
            crop_w = int(w * random.uniform(0.8, 1.0))
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            image = image[top:top+crop_h, left:left+crop_w]
            image = cv2.resize(image, (w, h))
        
        # Perspective transform
        if config.get('perspective', 0) > 0:
            h, w = image.shape[:2]
            scale = config['perspective'] * 0.1
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([
                [random.uniform(0, w*scale), random.uniform(0, h*scale)],
                [random.uniform(w*(1-scale), w), random.uniform(0, h*scale)],
                [random.uniform(0, w*scale), random.uniform(h*(1-scale), h)],
                [random.uniform(w*(1-scale), w), random.uniform(h*(1-scale), h)]
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image
    
    def stop(self):
        """Stop augmentation"""
        self.should_stop = True


class ImageViewer(QLabel):
    """Custom image viewer widget"""
    
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                color: #e0e0e0;
            }
        """)
        self.setText(f"{title}\n\nNo image loaded")
        self.current_pixmap = None
        
    def set_image(self, image: np.ndarray):
        """Set image from numpy array"""
        if image is None:
            self.setText(f"{self.title}\n\nNo image loaded")
            return
        
        h, w = image.shape[:2]
        bytes_per_line = 3 * w
        
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.width() - 20, 
            self.height() - 20, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.current_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)


class AugmentationGUI(QMainWindow):
    """Main augmentation GUI application"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialise user interface"""
        self.setWindowTitle("Image Augmentation Studio")
        self.setGeometry(100, 100, 1600, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("Image Augmentation Studio")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; padding: 15px;")
        main_layout.addWidget(title)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Image viewers
        right_panel = self.create_viewer_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([500, 1000])
        main_layout.addWidget(splitter)
        
        # Apply stylesheet
        self.apply_stylesheet()
        
    def create_control_panel(self) -> QWidget:
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # File selection
        file_group = QGroupBox("Image Selection")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        btn_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_single_image)
        btn_layout.addWidget(self.load_image_btn)
        
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        btn_layout.addWidget(self.load_folder_btn)
        file_layout.addLayout(btn_layout)
        
        # Image list
        self.image_list_widget = QListWidget()
        self.image_list_widget.setMaximumHeight(150)
        self.image_list_widget.currentRowChanged.connect(self.on_image_selected)
        file_layout.addWidget(self.image_list_widget)
        
        layout.addWidget(file_group)
        
        # Augmentation settings
        aug_scroll = QScrollArea()
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setMaximumHeight(400)
        
        aug_widget = QWidget()
        aug_layout = QVBoxLayout()
        aug_widget.setLayout(aug_layout)
        
        # Geometric transformations
        geo_group = QGroupBox("Geometric Transformations")
        geo_layout = QGridLayout()
        geo_group.setLayout(geo_layout)
        
        # Horizontal flip
        self.h_flip_check = QCheckBox("Horizontal Flip")
        self.h_flip_check.stateChanged.connect(self.update_preview)
        geo_layout.addWidget(self.h_flip_check, 0, 0, 1, 2)
        
        # Vertical flip
        self.v_flip_check = QCheckBox("Vertical Flip")
        self.v_flip_check.stateChanged.connect(self.update_preview)
        geo_layout.addWidget(self.v_flip_check, 1, 0, 1, 2)
        
        # Rotation
        geo_layout.addWidget(QLabel("Rotation (degrees):"), 2, 0)
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(0, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.update_preview)
        geo_layout.addWidget(self.rotation_slider, 2, 1)
        self.rotation_label = QLabel("0°")
        geo_layout.addWidget(self.rotation_label, 2, 2)
        
        # Random crop
        self.random_crop_check = QCheckBox("Random Crop (80-100%)")
        self.random_crop_check.stateChanged.connect(self.update_preview)
        geo_layout.addWidget(self.random_crop_check, 3, 0, 1, 2)
        
        # Perspective
        geo_layout.addWidget(QLabel("Perspective:"), 4, 0)
        self.perspective_slider = QSlider(Qt.Horizontal)
        self.perspective_slider.setRange(0, 10)
        self.perspective_slider.setValue(0)
        self.perspective_slider.valueChanged.connect(self.update_preview)
        geo_layout.addWidget(self.perspective_slider, 4, 1)
        self.perspective_label = QLabel("0")
        geo_layout.addWidget(self.perspective_label, 4, 2)
        
        aug_layout.addWidget(geo_group)
        
        # Color adjustments
        color_group = QGroupBox("Colour Adjustments")
        color_layout = QGridLayout()
        color_group.setLayout(color_layout)
        
        # Brightness
        color_layout.addWidget(QLabel("Brightness:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_preview)
        color_layout.addWidget(self.brightness_slider, 0, 1)
        self.brightness_label = QLabel("0%")
        color_layout.addWidget(self.brightness_label, 0, 2)
        
        # Contrast
        color_layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.update_preview)
        color_layout.addWidget(self.contrast_slider, 1, 1)
        self.contrast_label = QLabel("0%")
        color_layout.addWidget(self.contrast_label, 1, 2)
        
        # Saturation
        color_layout.addWidget(QLabel("Saturation:"), 2, 0)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(self.update_preview)
        color_layout.addWidget(self.saturation_slider, 2, 1)
        self.saturation_label = QLabel("0%")
        color_layout.addWidget(self.saturation_label, 2, 2)
        
        # Hue
        color_layout.addWidget(QLabel("Hue Shift:"), 3, 0)
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(0, 180)
        self.hue_slider.setValue(0)
        self.hue_slider.valueChanged.connect(self.update_preview)
        color_layout.addWidget(self.hue_slider, 3, 1)
        self.hue_label = QLabel("0°")
        color_layout.addWidget(self.hue_label, 3, 2)
        
        aug_layout.addWidget(color_group)
        
        # Effects
        effects_group = QGroupBox("Effects")
        effects_layout = QGridLayout()
        effects_group.setLayout(effects_layout)
        
        # Blur
        effects_layout.addWidget(QLabel("Gaussian Blur:"), 0, 0)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 50)
        self.blur_slider.setValue(0)
        self.blur_slider.valueChanged.connect(self.update_preview)
        effects_layout.addWidget(self.blur_slider, 0, 1)
        self.blur_label = QLabel("0")
        effects_layout.addWidget(self.blur_label, 0, 2)
        
        # Noise
        effects_layout.addWidget(QLabel("Noise:"), 1, 0)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(0)
        self.noise_slider.valueChanged.connect(self.update_preview)
        effects_layout.addWidget(self.noise_slider, 1, 1)
        self.noise_label = QLabel("0")
        effects_layout.addWidget(self.noise_label, 1, 2)
        
        aug_layout.addWidget(effects_group)
        
        aug_scroll.setWidget(aug_widget)
        layout.addWidget(aug_scroll)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        action_group.setLayout(action_layout)
        
        self.preview_btn = QPushButton("🔄 Generate Preview")
        self.preview_btn.clicked.connect(self.update_preview)
        self.preview_btn.setStyleSheet("background-color: #0d7377; padding: 10px;")
        action_layout.addWidget(self.preview_btn)
        
        self.reset_btn = QPushButton("↺ Reset All")
        self.reset_btn.clicked.connect(self.reset_parameters)
        self.reset_btn.setStyleSheet("background-color: #f39c12; padding: 10px;")
        action_layout.addWidget(self.reset_btn)
        
        self.save_btn = QPushButton("💾 Save Augmented Image")
        self.save_btn.clicked.connect(self.save_augmented_image)
        self.save_btn.setStyleSheet("background-color: #27ae60; padding: 10px;")
        action_layout.addWidget(self.save_btn)
        
        # Batch processing
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Augmentations per image:"))
        self.num_aug_spin = QSpinBox()
        self.num_aug_spin.setRange(1, 20)
        self.num_aug_spin.setValue(5)
        batch_layout.addWidget(self.num_aug_spin)
        action_layout.addLayout(batch_layout)
        
        self.save_original_check = QCheckBox("Save original images")
        self.save_original_check.setChecked(True)
        action_layout.addWidget(self.save_original_check)
        
        self.batch_btn = QPushButton("⚡ Batch Process All Images")
        self.batch_btn.clicked.connect(self.batch_process)
        self.batch_btn.setStyleSheet("background-color: #8e44ad; padding: 10px;")
        action_layout.addWidget(self.batch_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)
        
        self.stop_btn = QPushButton("⏹ Stop Processing")
        self.stop_btn.clicked.connect(self.stop_batch_process)
        self.stop_btn.setVisible(False)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; padding: 10px;")
        action_layout.addWidget(self.stop_btn)
        
        layout.addWidget(action_group)
        
        # Add stretch
        layout.addStretch()
        
        return panel
    
    def create_viewer_panel(self) -> QWidget:
        """Create image viewer panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Viewers
        viewers_layout = QHBoxLayout()
        
        self.original_viewer = ImageViewer("Original Image")
        viewers_layout.addWidget(self.original_viewer)
        
        self.augmented_viewer = ImageViewer("Augmented Image")
        viewers_layout.addWidget(self.augmented_viewer)
        
        layout.addLayout(viewers_layout)
        
        return panel
    
    def apply_stylesheet(self):
        """Apply dark theme stylesheet"""
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
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                color: #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: none;
                border-radius: 3px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
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
            QSlider::groove:horizontal {
                border: 1px solid #3a3a3a;
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d7377;
                border: 1px solid #0d7377;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #0e8a8f;
            }
            QSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
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
            QScrollArea {
                border: none;
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
        """)
    
    def load_single_image(self):
        """Load a single image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_list = [file_path]
            self.current_index = 0
            self.update_image_list()
            self.load_image(file_path)
    
    def load_folder(self):
        """Load all images from a folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        
        if folder_path:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            self.image_list = []
            
            for ext in extensions:
                self.image_list.extend(Path(folder_path).glob(f"*{ext}"))
                self.image_list.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            self.image_list = [str(p) for p in self.image_list]
            self.image_list.sort()
            
            if self.image_list:
                self.current_index = 0
                self.update_image_list()
                self.load_image(self.image_list[0])
            else:
                QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
    
    def update_image_list(self):
        """Update the image list widget"""
        self.image_list_widget.clear()
        for image_path in self.image_list:
            self.image_list_widget.addItem(os.path.basename(image_path))
        
        if self.image_list:
            self.image_list_widget.setCurrentRow(self.current_index)
    
    def on_image_selected(self, index: int):
        """Handle image selection from list"""
        if 0 <= index < len(self.image_list):
            self.current_index = index
            self.load_image(self.image_list[index])
    
    def load_image(self, image_path: str):
        """Load and display image"""
        self.current_image_path = image_path
        image = cv2.imread(image_path)
        
        if image is None:
            QMessageBox.warning(self, "Error", f"Failed to load image: {image_path}")
            return
        
        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_viewer.set_image(self.current_image)
        self.update_preview()
    
    def get_augmentation_config(self) -> dict:
        """Get current augmentation configuration"""
        return {
            'horizontal_flip': self.h_flip_check.isChecked(),
            'vertical_flip': self.v_flip_check.isChecked(),
            'rotation': self.rotation_slider.value(),
            'brightness': self.brightness_slider.value() / 100.0,
            'contrast': self.contrast_slider.value() / 100.0,
            'saturation': self.saturation_slider.value() / 100.0,
            'hue': self.hue_slider.value(),
            'blur': self.blur_slider.value() / 10.0,
            'noise': self.noise_slider.value() / 100.0,
            'random_crop': self.random_crop_check.isChecked(),
            'perspective': self.perspective_slider.value(),
            'save_original': self.save_original_check.isChecked(),
        }
    
    def update_preview(self):
        """Update augmented image preview"""
        if self.current_image is None:
            return
        
        # Update labels
        self.rotation_label.setText(f"{self.rotation_slider.value()}°")
        self.brightness_label.setText(f"{self.brightness_slider.value()}%")
        self.contrast_label.setText(f"{self.contrast_slider.value()}%")
        self.saturation_label.setText(f"{self.saturation_slider.value()}%")
        self.hue_label.setText(f"{self.hue_slider.value()}°")
        self.blur_label.setText(f"{self.blur_slider.value() / 10.0:.1f}")
        self.noise_label.setText(f"{self.noise_slider.value()}%")
        self.perspective_label.setText(f"{self.perspective_slider.value()}")
        
        # Apply augmentations
        config = self.get_augmentation_config()
        worker = AugmentationWorker([], "", config, 1)
        augmented = worker.apply_augmentations(self.current_image.copy(), config)
        
        self.augmented_viewer.set_image(augmented)
    
    def reset_parameters(self):
        """Reset all augmentation parameters"""
        self.h_flip_check.setChecked(False)
        self.v_flip_check.setChecked(False)
        self.rotation_slider.setValue(0)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.hue_slider.setValue(0)
        self.blur_slider.setValue(0)
        self.noise_slider.setValue(0)
        self.random_crop_check.setChecked(False)
        self.perspective_slider.setValue(0)
        self.update_preview()
    
    def save_augmented_image(self):
        """Save the current augmented image"""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Augmented Image",
            "",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        
        if file_path:
            config = self.get_augmentation_config()
            worker = AugmentationWorker([], "", config, 1)
            augmented = worker.apply_augmentations(self.current_image.copy(), config)
            
            cv2.imwrite(file_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Success", f"Image saved to: {file_path}")
    
    def batch_process(self):
        """Start batch processing"""
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        
        if not output_dir:
            return
        
        config = self.get_augmentation_config()
        num_augmentations = self.num_aug_spin.value()
        
        # Show progress UI
        self.progress_bar.setVisible(True)
        self.stop_btn.setVisible(True)
        self.batch_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(self.image_list))
        self.progress_bar.setValue(0)
        
        # Create and start worker
        self.worker = AugmentationWorker(self.image_list, output_dir, config, num_augmentations)
        self.worker.progress_update.connect(self.on_batch_progress)
        self.worker.augmentation_complete.connect(self.on_batch_complete)
        self.worker.augmentation_error.connect(self.on_batch_error)
        self.worker.start()
    
    def stop_batch_process(self):
        """Stop batch processing"""
        if self.worker:
            self.worker.stop()
    
    def on_batch_progress(self, current: int, total: int, filename: str):
        """Handle batch progress update"""
        self.progress_bar.setValue(current)
        self.setWindowTitle(f"Processing: {filename} ({current}/{total})")
    
    def on_batch_complete(self, total_images: int):
        """Handle batch completion"""
        self.progress_bar.setVisible(False)
        self.stop_btn.setVisible(False)
        self.batch_btn.setEnabled(True)
        self.setWindowTitle("Image Augmentation Studio")
        
        QMessageBox.information(
            self,
            "Complete",
            f"Batch processing complete!\n{total_images} images generated."
        )
    
    def on_batch_error(self, error: str):
        """Handle batch error"""
        self.progress_bar.setVisible(False)
        self.stop_btn.setVisible(False)
        self.batch_btn.setEnabled(True)
        self.setWindowTitle("Image Augmentation Studio")
        
        QMessageBox.critical(self, "Error", f"Batch processing error: {error}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = AugmentationGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

