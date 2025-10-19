# Deep SVDD Training

Deep SVDD is for anomaly detection - it learns what "normal" looks like and can spot when something is different. You only need normal samples to train it, which is handy when you don't have many anomaly examples.

## Quick Start

### GUI Version (easier)
```bash
# Install requirements
pip install -r ../../requirements/requirements_deep_svdd.txt

# Launch the GUI
python deep_svdd_gui.py
```

### Script Version
```bash
# Install requirements
pip install -r ../../requirements/requirements_deep_svdd.txt

# Run the script
python One_Class_Classification.py
```

## How It Works

- Learns a "hypersphere" in feature space around normal data
- Only needs normal/positive samples for training
- Uses ResNet to extract features
- Spots anything that doesn't fit the normal pattern

## Good For

- Finding defects in manufacturing
- Detecting unusual behavior in security systems
- Medical anomaly detection
- Quality control
- Anywhere you need to spot "weird" stuff

## GUI Features

- Pick your normal samples
- Adjust training settings
- Watch training progress
- See what the model learned
- Save trained models

## Training Process

1. Collect normal samples (the more the better)
2. ResNet extracts features from images
3. Model learns what normal features look like
4. Use it to detect anything that doesn't fit

## Settings

You can adjust:
- `embedding_dim`: Feature size (default: 512)
- `batch_size`: Training batch size (default: 16)
- `num_epochs`: Training epochs (default: 20)
- `lr`: Learning rate (default: 1e-4)
- `data_dir`: Where your normal samples are
