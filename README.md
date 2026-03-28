# EEG-Based Dataset Explicitly Targets the Transitions between Sitting and Standing for Exploring Neural Activation Patterns in Motor Imagery and Execution

A framework for EEG-based Motor Execution (ME) and Motor Imagery (MI) classfication using using the dataset targeting sit-to-stand and stand-to-sit transitions, recorded from 22 healthy participants. Three deep learning models — CTNet, EEGNet, and TCANet — were benchmarked using leave-one-subject-out cross-validation.

---

## Dependencies

- Python>=3.8.0
- numpy==2.0.2
- scipy==1.13.1
- scikit-learn==1.6.1
- mne==1.8.0
- pandas==2.3.3
- matplotlib==3.9.4
- seaborn==0.13.2
- tqdm==4.67.1
- torch==2.6.0
- joblib==1.5.2
- braindecode==1.3.2
---


## Project Structure

```
.
├── networks/
│   ├── __init__.py          # Exports EEGNet
│   ├── EEGNet.py            # EEGNet model
│   └── EEGTransformer.py    # TCANet / EEGTransformer model
├── utils/
│   ├── layers.py            # Custom PyTorch layers
│   ├── trainer.py           # Trainer class, EarlyStopping, DataLoader utils
│   ├── utils.py             # Data loading, logging helpers
├── train_DL_independent.py  # Main training script
└── results.py               # Results aggregation and visualization
```

---

## Usage

```bash
python train_DL_independent.py \
  --model EEGNet \
  --exp ME \
  --task sit_std \
  --win 2.0 \
  --log log_raw \
  --GPU 0
```

**Arguments:**

| Argument | Description |
|---|---|
| `--model` | Model name: `EEGNet`, `CTNet`, or `TCANet` |
| `--exp` | Experiment type: `ME` or `MI` |
| `--task` | Task label: `sit_std` or `std_sit` |
| `--win` | EEG window duration in seconds |
| `--onset` | Event onset in seconds (default: `2`) |
| `--log` | Output directory for logs (default: `log_test`) |
| `--GPU` | GPU device ID (default: `0`) |


### License
Copyright &copy; 2026-All rights reserved by [INTERFACES (BRAIN lab @ IST, VISTEC, Thailand)](https://www.facebook.com/interfaces.brainvistec).
Distributed by an [Apache License 2.0](https://github.com/b5510546671/eeg_sit_stand/blob/main/LICENSE).