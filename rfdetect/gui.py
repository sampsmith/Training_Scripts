import os
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from detection.datasets.factory import build_datasets
from detection.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, collate_fn
from detection.utils.engine import train_one_epoch, evaluate
from detection.utils.coco_eval import coco_evaluate

import torch
import torchvision
from torch.utils.data import DataLoader


def build_model(num_classes: int, pretrained_weights: str = "COCO", trainable_backbone_layers: int = 3,
                box_score_thresh: float = 0.05, box_nms_thresh: float = 0.5, box_detections_per_img: int = 300):
    """
    Build Faster R-CNN model with pretrained weights.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained_weights: "COCO" (default, best for detection), "IMAGENET" (backbone only), or "NONE"
        trainable_backbone_layers: Number of backbone layers to fine-tune (0-5)
    """
    if pretrained_weights.upper() == "COCO":
        # Use COCO-pretrained Faster R-CNN (best for detection tasks)
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        backbone_weights_enum = None  # Backbone already pretrained in COCO weights
    elif pretrained_weights.upper() == "IMAGENET":
        # Use ImageNet-pretrained backbone only
        weights = None
        backbone_weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    elif pretrained_weights.upper() == "NONE":
        # No pretrained weights
        weights = None
        backbone_weights_enum = None
    else:
        # Default to COCO
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        backbone_weights_enum = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=backbone_weights_enum,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Replace classifier head for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Set detection thresholds
    model.roi_heads.score_thresh = box_score_thresh
    model.roi_heads.nms_thresh = box_nms_thresh
    model.roi_heads.detections_per_img = box_detections_per_img
    return model


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    epochs: int = 25
    batch_size: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 8
    lr_gamma: float = 0.1
    pretrained_weights: str = "COCO"
    trainable_backbone_layers: int = 3
    amp: bool = True
    eval_interval: int = 1
    save_interval: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Worker(QtCore.QObject):
    progress = QtCore.Signal(str)
    batch_progress = QtCore.Signal(int, int, dict)  # current, total, losses
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.progress.emit("Preparing datasets...")
            train_tf = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
            valid_tf = Compose([ToTensor()])
            train_ds, valid_ds, eval_type = build_datasets(self.cfg.data_dir, train_tf, valid_tf)

            train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                                      num_workers=4, collate_fn=collate_fn, pin_memory=True)
            valid_loader = DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False,
                                      num_workers=4, collate_fn=collate_fn, pin_memory=True)

            self.progress.emit("Building model...")
            model = build_model(
                num_classes=train_ds.num_classes,
                pretrained_weights=self.cfg.pretrained_weights,
                trainable_backbone_layers=self.cfg.trainable_backbone_layers,
            )
            device = torch.device(self.cfg.device)
            model.to(device)

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)
            scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

            best_ap = -1.0
            for epoch in range(1, self.cfg.epochs + 1):
                self.progress.emit(f"Epoch {epoch}/{self.cfg.epochs} - training...")
                
                def progress_callback(current, total, losses):
                    self.batch_progress.emit(current, total, losses)
                
                loss_stats = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, 
                                           progress_callback=progress_callback)
                lr_scheduler.step()

                if (epoch % self.cfg.save_interval) == 0:
                    path = os.path.join(self.cfg.output_dir, f"model_epoch_{epoch:03d}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    }, path)
                    self.progress.emit(f"Saved checkpoint: {path}")

                if (epoch % self.cfg.eval_interval) == 0:
                    self.progress.emit("Evaluating...")
                    outputs, image_ids = evaluate(model, valid_loader, device)
                    if eval_type == "coco":
                        _, cont_to_cat = train_ds.get_cat_mappings()
                        metrics, _ = coco_evaluate(valid_ds.get_coco_api(), outputs, image_ids, cont_to_cat)
                        ap = metrics.get("AP", 0.0)
                        self.progress.emit(f"Validation AP: {ap:.4f}")
                    else:
                        ap = float(torch.mean(torch.cat([o["scores"] for o in outputs])[:100]).item()) if len(outputs) else 0.0
                        self.progress.emit(f"Validation score(mean top100): {ap:.4f}")
                    if ap > best_ap:
                        best_ap = ap
                        best_path = os.path.join(self.cfg.output_dir, "model_best.pth")
                        torch.save({
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "best_ap": best_ap,
                        }, best_path)
                        self.progress.emit(f"Saved best model: {best_path}")

            self.progress.emit("Training complete.")
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboflow Faster R-CNN Trainer")
        self.resize(900, 600)

        # State
        self.dataset_dir: Optional[str] = None

        # Widgets
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        # Dataset import
        grp_data = QtWidgets.QGroupBox("Dataset")
        data_layout = QtWidgets.QHBoxLayout(grp_data)
        self.txt_data = QtWidgets.QLineEdit()
        self.txt_data.setPlaceholderText("/abs/path/to/roboflow_coco_root (contains train/, valid/)")
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self.on_browse)
        btn_import_zip = QtWidgets.QPushButton("Import Zip…")
        btn_import_zip.clicked.connect(self.on_import_zip)
        data_layout.addWidget(self.txt_data)
        data_layout.addWidget(btn_browse)
        data_layout.addWidget(btn_import_zip)

        # Output dir
        grp_out = QtWidgets.QGroupBox("Output")
        out_layout = QtWidgets.QHBoxLayout(grp_out)
        self.txt_out = QtWidgets.QLineEdit()
        self.txt_out.setPlaceholderText("/abs/path/to/outputs")
        btn_out = QtWidgets.QPushButton("Choose…")
        btn_out.clicked.connect(self.on_choose_out)
        out_layout.addWidget(self.txt_out)
        out_layout.addWidget(btn_out)

        # Hyperparameters
        grp_hp = QtWidgets.QGroupBox("Hyperparameters")
        form = QtWidgets.QFormLayout(grp_hp)
        self.sp_epochs = QtWidgets.QSpinBox(); self.sp_epochs.setRange(1, 1000); self.sp_epochs.setValue(25)
        self.sp_batch = QtWidgets.QSpinBox(); self.sp_batch.setRange(1, 64); self.sp_batch.setValue(4)
        self.d_lr = QtWidgets.QDoubleSpinBox(); self.d_lr.setDecimals(6); self.d_lr.setRange(1e-6, 1.0); self.d_lr.setValue(0.005)
        self.sp_step = QtWidgets.QSpinBox(); self.sp_step.setRange(1, 1000); self.sp_step.setValue(8)
        self.d_gamma = QtWidgets.QDoubleSpinBox(); self.d_gamma.setDecimals(3); self.d_gamma.setRange(0.001, 1.0); self.d_gamma.setValue(0.1)
        self.sp_tblr = QtWidgets.QSpinBox(); self.sp_tblr.setRange(0,5); self.sp_tblr.setValue(3)
        self.cmb_pretrained = QtWidgets.QComboBox(); self.cmb_pretrained.addItems(["COCO", "IMAGENET", "NONE"])
        self.chk_amp = QtWidgets.QCheckBox(); self.chk_amp.setChecked(True)
        self.cmb_device = QtWidgets.QComboBox(); self.cmb_device.addItems(["cuda", "cpu"])
        if not torch.cuda.is_available():
            self.cmb_device.setCurrentText("cpu")
        form.addRow("Epochs", self.sp_epochs)
        form.addRow("Batch Size", self.sp_batch)
        form.addRow("Learning Rate", self.d_lr)
        form.addRow("LR Step Size", self.sp_step)
        form.addRow("LR Gamma", self.d_gamma)
        form.addRow("Pretrained Weights", self.cmb_pretrained)
        form.addRow("Trainable Backbone Layers", self.sp_tblr)
        form.addRow("Mixed Precision (AMP)", self.chk_amp)
        form.addRow("Device", self.cmb_device)

        # Controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_train = QtWidgets.QPushButton("Start Training")
        self.btn_train.clicked.connect(self.on_start)
        self.btn_train.setEnabled(True)
        ctrl_layout.addWidget(self.btn_train)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_progress = QtWidgets.QLabel("")
        self.lbl_progress.setVisible(False)

        # Log
        self.txt_log = QtWidgets.QPlainTextEdit(); self.txt_log.setReadOnly(True)

        layout.addWidget(grp_data)
        layout.addWidget(grp_out)
        layout.addWidget(grp_hp)
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_progress)
        layout.addWidget(self.txt_log, 1)

        # Threading
        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[Worker] = None

    def on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset root")
        if d:
            self.txt_data.setText(d)

    def on_choose_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.txt_out.setText(d)

    def on_import_zip(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Roboflow COCO zip", filter="Zip Files (*.zip)")
        if not path:
            return
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select extract destination")
        if not target_dir:
            return
        self.append_log(f"Extracting {path} → {target_dir} ...")
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(target_dir)
        self.append_log("Extract complete.")
        self.txt_data.setText(target_dir)

    def append_log(self, text: str):
        self.txt_log.appendPlainText(text)

    def on_start(self):
        data_dir = self.txt_data.text().strip()
        output_dir = self.txt_out.text().strip()
        if not data_dir or not os.path.isdir(data_dir):
            self.append_log("Please select a valid dataset directory.")
            return
        if not output_dir:
            self.append_log("Please select an output directory.")
            return

        cfg = TrainConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=self.sp_epochs.value(),
            batch_size=self.sp_batch.value(),
            lr=float(self.d_lr.value()),
            lr_step_size=self.sp_step.value(),
            lr_gamma=float(self.d_gamma.value()),
            pretrained_weights=self.cmb_pretrained.currentText(),
            trainable_backbone_layers=self.sp_tblr.value(),
            amp=self.chk_amp.isChecked(),
            device=self.cmb_device.currentText(),
        )

        self.btn_train.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_progress.setVisible(True)
        self.progress_bar.setValue(0)
        self.append_log("Starting training...")

        self.thread = QtCore.QThread()
        self.worker = Worker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.append_log)
        self.worker.batch_progress.connect(self.on_batch_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_batch_progress(self, current: int, total: int, losses: dict):
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        
        loss_str = ", ".join([f"{k}:{v:.4f}" for k, v in losses.items()])
        self.lbl_progress.setText(f"Batch {current}/{total} - {loss_str}")

    def on_finished(self):
        self.append_log("Done.")
        self.btn_train.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)

    def on_failed(self, msg: str):
        self.append_log(f"Error: {msg}")
        self.btn_train.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setVisible(False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


