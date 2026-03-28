import os
import random
import argparse
import numpy as np
import os.path as op

import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from networks import EEGNet
from networks.EEGTransformer import EEGTransformer
from braindecode.models import CTNet
from utils.trainer import Trainer, CustomDataset, StratifiedBatchSampler
from utils.utils import write_log, load_data

# Suppress MNE and PyTorch warnings
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Environment settings for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(SEED)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id):
    """Initialize NumPy and Python RNGs in each worker process for DataLoader."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Train_DL_independent(model_name, experiment, task, window_duration, onset=2, dataset_path=op.join('..', 'data', 'raw'), log_path="log_test"):
    
    X, y, class_names = load_data(experiment=experiment, task=task, window_duration=window_duration, onset=onset, dataset_path=dataset_path)

    for subject, (X_test, y_test) in enumerate(zip(X, y)):
        print(f"Testing subject: S{subject+1:02}")

        # Merge all other subjects for training
        X_tr = np.concatenate([X_rem for i, X_rem in enumerate(X) if i != subject], axis=0)
        y_tr = np.concatenate([y_rem for j, y_rem in enumerate(y) if j != subject], axis=0)

        # Convert test set to tensors
        X_test = torch.unsqueeze(torch.tensor(X_test, dtype=torch.float32), dim=1)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # 5-fold Stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            print(f"============== {model_name} Subject: {subject+1:02} Fold: {fold+1:02} ==============")

            # Split into train/val sets
            X_train, y_train = X_tr[train_idx], y_tr[train_idx]
            X_val, y_val = X_tr[val_idx], y_tr[val_idx]

            # Dimensions
            n_channel, n_timepoint = X_train.shape[1], X_train.shape[2]
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

            # Convert to tensors and add channel dimension
            X_train = torch.unsqueeze(torch.tensor(X_train, dtype=torch.float32), dim=1)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_val = torch.unsqueeze(torch.tensor(X_val, dtype=torch.float32), dim=1)
            y_val = torch.tensor(y_val, dtype=torch.long)

            print(f"TRAIN \t\t{X_train.shape} {y_train.shape}")
            print(f"VALIDATE \t{X_val.shape} {y_val.shape}")
            print(f"TEST \t\t{X_test.shape} {y_test.shape}")
            print(f"Class weight: \t{class_weights}")

            # Create PyTorch Datasets
            train_data = CustomDataset(X_train, y_train)
            val_data = CustomDataset(X_val, y_val)

            # Training hyperparameters
            N_CLASS = 2
            BATCH_SIZE = 8
            LEARNING_RATE = 1e-3
            NUM_EPOCH = 200
            PATIENCE = 10
            S_FREQ = 250

            # Create data loaders with stratified batching
            train_loader = DataLoader(dataset=train_data,
                                      batch_sampler=StratifiedBatchSampler(y_train, batch_size=BATCH_SIZE, shuffle=True),
                                      worker_init_fn=seed_worker,
                                      generator=torch.Generator().manual_seed(SEED),
                                      num_workers=0)

            val_loader = DataLoader(dataset=val_data,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    worker_init_fn=seed_worker,
                                    generator=torch.Generator().manual_seed(SEED),
                                    num_workers=0)

            # Initialize model
            if model_name == "EEGNet":
                model = EEGNet(n_channel=n_channel, n_timepoint=n_timepoint, n_class=N_CLASS).to(device)
            elif model_name == "CTNet":
                model = CTNet(n_times=n_timepoint, n_chans=n_channel, n_outputs=N_CLASS, sfreq=S_FREQ, input_window_seconds=n_timepoint/S_FREQ).to(device)
            elif model_name == "TCANet":
                model = EEGTransformer(n_chans=n_channel, n_classes=N_CLASS, n_times=n_timepoint, pooling_size=56).to(device)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            print(f"Model: {summary(model)}")

            # Define output paths
            log_dir = os.path.join(log_path, f"{model_name}_results", experiment.upper(), task, f"{window_duration}s")
            os.makedirs(log_dir, exist_ok=True)
            model_path = os.path.join(log_dir, f"model_{model_name}_S{subject+1:02}_f{fold+1:02}.pt")

            # Define loss function and optimizer
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Train
            trainer = Trainer(model=model,
                              batch_size=BATCH_SIZE,
                              n_epoch=NUM_EPOCH,
                              n_class=N_CLASS,
                              patience=PATIENCE,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              directory=model_path)

            tracker = trainer.train(train_loader, val_loader)

            # Evaluate on the held-out subject
            predictions, evaluation = trainer.eval_model(model, X_test, y_test, class_names)
            evaluation.update({"fold": fold+1})

            # Save loss curves
            loss_path = os.path.join(log_dir, f"loss_{model_name}_S{subject+1:02}_f{fold+1:02}.npz")
            np.savez(loss_path,
                     train_tracker=np.array(tracker["train_tracker"]),
                     val_tracker=np.array(tracker["val_tracker"]),
                     training_time_tracker=np.array(tracker["training_time_tracker"]))

            # Append evaluation to CSV
            report_path = os.path.join(log_dir, f"report_{model_name}_S{subject+1:02}.csv")
            if fold == 0:
                write_log(report_path, data=list(evaluation.keys()), mode="w")
            write_log(report_path, data=list(evaluation.values()), mode="a")

            # Save predictions
            result_path = os.path.join(log_dir, f"result_{model_name}_S{subject+1:02}_f{fold+1:02}.npz")
            np.savez(result_path,
                     y_true=np.array(predictions["y_true"]),
                     y_pred=np.array(predictions["y_pred"]))
            

        print(f"------------------------- S{subject+1:02} Done --------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG model on ME or MI datasets.")
    parser.add_argument("--model", type=str, required=True, help="Model name to train (e.g., 'EEGNet').")
    parser.add_argument("--exp", type=str, choices=["ME", "MI"], required=True, help="Experiment type: 'ME' for motor execution or 'MI' for motor imagery.")
    parser.add_argument("--task", type=str, required=True, help="Task label. Either 'sit_std' or 'std_sit'.")
    parser.add_argument("--win", type=float, required=True, help="EEG window duration in seconds.")
    parser.add_argument("--onset", type=float, help="Event onset in seconds (default: 2).")
    parser.add_argument("--path", type=float, help="Dataset path (default: /mount/MI_EEG_database/data/raw).")
    parser.add_argument("--log", type=str, default="log_test", help="Output path for logs and results (default: log_test).")
    parser.add_argument("--GPU", type=int, default=0, help="GPU ID to use (default: 0).")
    args = parser.parse_args()

    torch.cuda.set_device(args.GPU)

    Train_DL_independent(
        model_name=args.model,
        experiment=args.exp,
        task=args.task,
        window_duration=args.win,
        log_path=args.log,
    )

    
"""
python train_DL_independent.py --model EEGNet --exp ME --task sit_std --win 2 --log log_raw --GPU 0 && python train_DL_independent.py --model EEGNet --exp ME --task std_sit --win 2 --log log_raw --GPU 0
python train_DL_independent.py --model EEGNet --exp ME --task sit_std --win 1 --log log_raw --GPU 0 && python train_DL_independent.py --model EEGNet --exp ME --task std_sit --win 1 --log log_raw --GPU 0
    
python train_DL_independent.py --model EEGNet --exp MI --task sit_std --win 5 --log log_raw --GPU 1 && python train_DL_independent.py --model EEGNet --exp MI --task std_sit --win 5 --log log_raw --GPU 1
python train_DL_independent.py --model EEGNet --exp MI --task sit_std --win 4 --log log_raw --GPU 1 && python train_DL_independent.py --model EEGNet --exp MI --task std_sit --win 4 --log log_raw --GPU 1
python train_DL_independent.py --model EEGNet --exp MI --task sit_std --win 3 --log log_raw --GPU 2 && python train_DL_independent.py --model EEGNet --exp MI --task std_sit --win 3 --log log_raw --GPU 2
python train_DL_independent.py --model EEGNet --exp MI --task sit_std --win 2 --log log_raw --GPU 2 && python train_DL_independent.py --model EEGNet --exp MI --task std_sit --win 2 --log log_raw --GPU 2
python train_DL_independent.py --model EEGNet --exp MI --task sit_std --win 1 --log log_raw --GPU 2 && python train_DL_independent.py --model EEGNet --exp MI --task std_sit --win 1 --log log_raw --GPU 2
"""

