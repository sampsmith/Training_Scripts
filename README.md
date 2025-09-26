## CLI: Train Faster R-CNN ResNet‑50 on Roboflow COCO (PyTorch)

Train a Faster R-CNN (ResNet‑50 FPN) model on a Roboflow COCO-format dataset using a single command-line interface.

### Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train_fasterrcnn.py \
  --data_dir /abs/path/to/datasets/my_coco \
  --output_dir /abs/path/to/outputs \
  --device cuda --amp
```

Install CLI (optional):
```bash
pip install -e .
rfdetect --help
rfdetect-gui
```

### Dataset layout (Roboflow COCO export)

```
/abs/path/to/datasets/my_coco/
  train/
    _annotations.coco.json
    *.jpg|*.png...
  valid/
    _annotations.coco.json
    *.jpg|*.png...
  test/                # optional
    _annotations.coco.json
    *.jpg|*.png...
```

### Full CLI reference

Paths
- `--data_dir` (str, required): Root containing `train/`, `valid/` (and optional `test/`).
- `--output_dir` (str, default: `outputs`): Where checkpoints and logs are saved.

Optimiser & scheduler
- `--epochs` (int, default: 25)
- `--batch_size` (int, default: 4)
- `--lr` (float, default: 0.005)
- `--momentum` (float, default: 0.9)
- `--weight_decay` (float, default: 0.0005)
- `--lr_step_size` (int, default: 8)
- `--lr_gamma` (float, default: 0.1)

Model/backbone
- `--backbone_weights` (`IMAGENET`|`NONE`, default: `IMAGENET`)
- `--trainable_backbone_layers` (0–5, default: 3)
- `--box_score_thresh` (float, default: 0.05)
- `--box_nms_thresh` (float, default: 0.5)
- `--box_detections_per_img` (int, default: 300)

Data loading & augmentation
- `--num_workers` (int, default: 4)
- `--flip_prob` (float, default: 0.5): Random horizontal flip probability (train only).

Training controls
- `--seed` (int, default: 42)
- `--device` (str, default: `cuda` if available else `cpu`)
- `--amp` (flag): Enable mixed precision training.
- `--grad_clip` (float, default: 0.0): Max global grad norm (0 disables).
- `--accumulation_steps` (int, default: 1): Gradient accumulation steps.

Evaluation
- `--eval_interval` (int, default: 1): Evaluate every N epochs on `valid/`.

Checkpointing
- `--resume` (str, default: empty): Path to checkpoint to resume from.
- `--save_interval` (int, default: 1): Save checkpoint every N epochs.

### Examples

Balanced defaults on GPU with AMP:
```bash
python train_fasterrcnn.py \
  --data_dir /abs/path/datasets/my_coco \
  --output_dir /abs/path/outputs \
  --epochs 25 --batch_size 4 --lr 0.005 \
  --device cuda --amp
```

Longer schedule with smaller LR and heavier backbone fine-tuning:
```bash
python train_fasterrcnn.py \
  --data_dir /abs/path/datasets/my_coco \
  --output_dir /abs/path/outputs_long \
  --epochs 50 --batch_size 4 --lr 0.0025 \
  --trainable_backbone_layers 5 --lr_step_size 12 --lr_gamma 0.1 \
  --device cuda --amp
```

Resume from checkpoint and evaluate every 2 epochs:
```bash
python train_fasterrcnn.py \
  --data_dir /abs/path/datasets/my_coco \
  --output_dir /abs/path/outputs \
  --resume /abs/path/outputs/model_epoch_010.pth \
  --eval_interval 2 --device cuda
```

CPU training (small experiments):
```bash
python train_fasterrcnn.py --data_dir /abs/path/datasets/my_coco --device cpu
```

CLI subcommands (after `pip install -e .`):
- Train:
```bash
rfdetect train --data_dir /abs/path/datasets/my_coco --output_dir /abs/path/outputs --device cuda --amp
```
- Evaluate:
```bash
rfdetect eval --data_dir /abs/path/datasets/my_coco --checkpoint /abs/path/outputs/model_best.pth --device cuda
```
- Predict on a folder of images:
```bash
rfdetect predict --checkpoint /abs/path/outputs/model_best.pth \
  --images_dir /abs/path/images \
  --output_dir /abs/path/preds --score_thresh 0.5 --device cuda
```

### Outputs
- Checkpoints: `model_epoch_XXX.pth` per `--save_interval` and `model_best.pth` (best validation AP).
- Console log prints epoch losses, LR, and latest/best AP.

### Troubleshooting
- Ensure you pass absolute paths for `--data_dir` and `--output_dir`.
- If images aren’t found, verify the Roboflow export structure and filenames inside `_annotations.coco.json` match files on disk.
- If CUDA OOM occurs: reduce `--batch_size`, increase `--accumulation_steps`, or disable `--amp` as a test.
- If COCO evaluation shows AP=0: confirm categories are present in `valid/` and boxes aren’t all filtered by thresholds.

### Desktop app (GUI)
After `pip install -e .`, launch the GUI:
```bash
rfdetect-gui
```
Features:
- Select Roboflow COCO dataset folder (or import a ZIP to extract).
- Choose output directory, epochs, batch size, LR, scheduler, AMP, and device.
- Training runs in background; progress and validation AP are shown in the log.


