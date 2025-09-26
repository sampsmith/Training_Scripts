import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torch.utils.data import DataLoader
from PySide6 import QtCore, QtWidgets

from detection.datasets.factory import build_datasets
from detection.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, collate_fn
from detection.utils.coco_eval import coco_evaluate


def build_model(num_classes: int, pretrained_weights: str = "COCO", trainable_backbone_layers: int = 3):
    if pretrained_weights.upper() == "COCO":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        backbone_weights_enum = None
    elif pretrained_weights.upper() == "IMAGENET":
        weights = None
        backbone_weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
        backbone_weights_enum = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=backbone_weights_enum,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


@dataclass
class AutoConfig:
    data_dir: str
    output_dir: str
    epochs: int = 30
    patience: int = 5
    pretrained_weights: str = "COCO"
    trainable_backbone_layers: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    grad_clip: float = 0.0
    target_effective_bsz: int = 16


class AutoWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    batch_progress = QtCore.Signal(int, int, dict)
    epoch_progress = QtCore.Signal(int, int)
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, cfg: AutoConfig):
        super().__init__()
        self.cfg = cfg

    def _try_batch_size(self, model, train_ds, batch_size: int) -> bool:
        try:
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
            device = torch.device(self.cfg.device)
            model.to(device)
            images, targets = next(iter(loader))
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast(enabled=self.cfg.use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize() if device.type == "cuda" else None
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    def _find_largest_batch(self, model, train_ds) -> (int, int):
        candidates = [32, 16, 8, 4, 2, 1]
        for bs in candidates:
            self.log.emit(f"Trying batch size {bs}...")
            if self._try_batch_size(model, train_ds, bs):
                self.log.emit(f"Selected batch size {bs}")
                accum = max(1, self.cfg.target_effective_bsz // bs)
                return bs, accum
        return 1, self.cfg.target_effective_bsz

    def _lr_range_test(self, model, train_loader, device) -> float:
        min_lr = 1e-6
        max_lr = 1e-1
        iters = min(150, len(train_loader))
        gamma = (max_lr / min_lr) ** (1.0 / max(1, iters - 1))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=min_lr, momentum=0.9, weight_decay=5e-4)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        lr = min_lr
        avg_loss = 0.0
        best_lr = lr
        best_slope = float("inf")
        beta = 0.98

        model.train()
        for step, (images, targets) in enumerate(train_loader):
            if step >= iters:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            for g in optimizer.param_groups:
                g["lr"] = lr
            with autocast(enabled=self.cfg.use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

            loss_item = loss.detach().item()
            avg_loss = beta * avg_loss + (1 - beta) * loss_item
            smooth = avg_loss / (1 - beta ** (step + 1))
            if step > 5:
                slope = smooth / lr
                if slope < best_slope:
                    best_slope = slope
                    best_lr = lr
            lr *= gamma
        self.log.emit(f"LR finder suggests ~{best_lr:.5f}")
        return float(max(1e-5, min(best_lr, 1e-2)))

    def _warmup_cosine(self, optimizer, steps_warmup: int, steps_total: int, base_lr: float):
        def lr_lambda(step):
            if step < steps_warmup:
                return float(step + 1) / float(max(1, steps_warmup))
            progress = (step - steps_warmup) / float(max(1, steps_total - steps_warmup))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    @QtCore.Slot()
    def run(self):
        try:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.log.emit("Loading dataset...")
            tf_train = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
            tf_valid = Compose([ToTensor()])
            train_ds, valid_ds, eval_type = build_datasets(self.cfg.data_dir, tf_train, tf_valid)

            model = build_model(train_ds.num_classes, pretrained_weights=self.cfg.pretrained_weights, trainable_backbone_layers=self.cfg.trainable_backbone_layers)
            device = torch.device(self.cfg.device)
            model.to(device)

            bs, accum = self._find_largest_batch(model, train_ds)
            self.log.emit(f"Batch size: {bs}, Accumulation: {accum}")

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
            valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

            lr = self._lr_range_test(model, train_loader, device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

            steps_total = self.cfg.epochs * max(1, len(train_loader) // max(1, accum))
            sched = self._warmup_cosine(optimizer, steps_warmup=max(10, len(train_loader)//5), steps_total=steps_total, base_lr=lr)
            scaler = GradScaler(enabled=self.cfg.use_amp)

            best_metric = -1.0
            best_path = os.path.join(self.cfg.output_dir, "model_best.pth")
            patience_counter = 0

            for epoch in range(1, self.cfg.epochs + 1):
                self.epoch_progress.emit(epoch, self.cfg.epochs)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                total_batches = len(train_loader)
                for step, (images, targets) in enumerate(train_loader):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with autocast(enabled=self.cfg.use_amp):
                        loss_dict = model(images, targets)
                        loss = sum(loss for loss in loss_dict.values())
                    if scaler.is_enabled():
                        scaler.scale(loss / accum).backward()
                    else:
                        (loss / accum).backward()

                    if ((step + 1) % accum) == 0:
                        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                            if scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                        if scaler.is_enabled():
                            scaler.step(optimizer); scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        sched.step()

                    losses_cpu = {k: float(v.detach().item()) for k, v in loss_dict.items()}
                    self.batch_progress.emit(step + 1, total_batches, losses_cpu)

                model.eval()
                with torch.no_grad():
                    outputs_all: List[Dict] = []
                    image_ids: List[int] = []
                    for images, targets in valid_loader:
                        images = [img.to(device) for img in images]
                        outputs = model(images)
                        outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]
                        outputs_all.extend(outputs)
                        image_ids.extend([int(t["image_id"].item()) for t in targets])

                if eval_type == "coco":
                    _, cont_to_cat = train_ds.get_cat_mappings()
                    metrics, _ = coco_evaluate(valid_ds.get_coco_api(), outputs_all, image_ids, cont_to_cat)
                    current = metrics.get("AP", 0.0)
                    self.log.emit(f"Epoch {epoch}: AP={current:.4f}")
                else:
                    current = float(torch.mean(torch.cat([o["scores"] for o in outputs_all])[:100]).item()) if len(outputs_all) else 0.0
                    self.log.emit(f"Epoch {epoch}: score(mean top100)={current:.4f}")

                ckpt_path = os.path.join(self.cfg.output_dir, f"model_epoch_{epoch:03d}.pth")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)

                if current > best_metric:
                    best_metric = current
                    patience_counter = 0
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_metric": best_metric,
                    }, best_path)
                    self.log.emit(f"Saved best: {best_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.patience:
                        self.log.emit("Early stopping")
                        break

            self.log.emit("AutoTrain complete.")
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))


class AutoWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roboflow AutoTrain (Faster R-CNN)")
        self.resize(900, 620)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        grp_data = QtWidgets.QGroupBox("Dataset")
        h = QtWidgets.QHBoxLayout(grp_data)
        self.txt_data = QtWidgets.QLineEdit(); self.txt_data.setPlaceholderText("/abs/path/to/dataset root")
        btn_browse = QtWidgets.QPushButton("Browse…"); btn_browse.clicked.connect(self.on_browse)
        h.addWidget(self.txt_data); h.addWidget(btn_browse)

        grp_out = QtWidgets.QGroupBox("Output")
        h2 = QtWidgets.QHBoxLayout(grp_out)
        self.txt_out = QtWidgets.QLineEdit(); self.txt_out.setPlaceholderText("/abs/path/to/outputs")
        btn_out = QtWidgets.QPushButton("Choose…"); btn_out.clicked.connect(self.on_choose_out)
        h2.addWidget(self.txt_out); h2.addWidget(btn_out)

        grp_hp = QtWidgets.QGroupBox("Auto settings")
        form = QtWidgets.QFormLayout(grp_hp)
        self.sp_epochs = QtWidgets.QSpinBox(); self.sp_epochs.setRange(1, 1000); self.sp_epochs.setValue(30)
        self.sp_patience = QtWidgets.QSpinBox(); self.sp_patience.setRange(1, 50); self.sp_patience.setValue(5)
        self.cmb_pretrained = QtWidgets.QComboBox(); self.cmb_pretrained.addItems(["COCO", "IMAGENET", "NONE"]) 
        self.sp_tblr = QtWidgets.QSpinBox(); self.sp_tblr.setRange(0,5); self.sp_tblr.setValue(3)
        self.sp_eff_bsz = QtWidgets.QSpinBox(); self.sp_eff_bsz.setRange(1, 128); self.sp_eff_bsz.setValue(16)
        self.chk_amp = QtWidgets.QCheckBox(); self.chk_amp.setChecked(True)
        self.cmb_device = QtWidgets.QComboBox(); self.cmb_device.addItems(["cuda", "cpu"]) 
        if not torch.cuda.is_available():
            self.cmb_device.setCurrentText("cpu")
        form.addRow("Epochs", self.sp_epochs)
        form.addRow("Early Stop Patience", self.sp_patience)
        form.addRow("Pretrained Weights", self.cmb_pretrained)
        form.addRow("Trainable Backbone Layers", self.sp_tblr)
        form.addRow("Target Effective Batch Size", self.sp_eff_bsz)
        form.addRow("Mixed Precision (AMP)", self.chk_amp)
        form.addRow("Device", self.cmb_device)

        self.btn_start = QtWidgets.QPushButton("Start AutoTrain")
        self.btn_start.clicked.connect(self.on_start)

        self.progress_bar = QtWidgets.QProgressBar(); self.progress_bar.setVisible(False)
        self.lbl_batch = QtWidgets.QLabel(""); self.lbl_batch.setVisible(False)
        self.txt_log = QtWidgets.QPlainTextEdit(); self.txt_log.setReadOnly(True)

        layout.addWidget(grp_data)
        layout.addWidget(grp_out)
        layout.addWidget(grp_hp)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_batch)
        layout.addWidget(self.txt_log, 1)

        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[AutoWorker] = None

    def on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset root")
        if d:
            self.txt_data.setText(d)

    def on_choose_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.txt_out.setText(d)

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

        cfg = AutoConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=self.sp_epochs.value(),
            patience=self.sp_patience.value(),
            pretrained_weights=self.cmb_pretrained.currentText(),
            trainable_backbone_layers=self.sp_tblr.value(),
            device=self.cmb_device.currentText(),
            target_effective_bsz=self.sp_eff_bsz.value(),
            use_amp=self.chk_amp.isChecked(),
        )

        self.btn_start.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.lbl_batch.setVisible(True)
        self.append_log("Starting AutoTrain...")

        self.thread = QtCore.QThread()
        self.worker = AutoWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.epoch_progress.connect(self.on_epoch_progress)
        self.worker.batch_progress.connect(self.on_batch_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_epoch_progress(self, current: int, total: int):
        self.append_log(f"Epoch {current}/{total}")
        self.progress_bar.setValue(0)

    def on_batch_progress(self, current: int, total: int, losses: Dict[str, float]):
        self.progress_bar.setValue(int((current/total) * 100))
        loss_str = ", ".join([f"{k}:{v:.4f}" for k, v in losses.items()])
        self.lbl_batch.setText(f"Batch {current}/{total} - {loss_str}")

    def on_finished(self):
        self.append_log("Done.")
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_batch.setVisible(False)

    def on_failed(self, msg: str):
        self.append_log(f"Error: {msg}")
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_batch.setVisible(False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = AutoWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


