import os
import os.path as op
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from torchmetrics.classification import MulticlassF1Score


# Reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

def seed_worker():
    np.random.seed(random_seed)
    random.seed(random_seed)

torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True

g = torch.Generator()
g.manual_seed(random_seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampling
    Provides equal representation of target classes in each batch.
    
    Args:
        y (array-like): Target class labels.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last batch if it's not a full batch.
    """
    def __init__(self, y, batch_size, shuffle=True, drop_last=False):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, "Labels must be 1D"
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = self._make_batches()

    def _make_batches(self):
        n_batches = int(np.floor(len(self.y) / self.batch_size)) if self.drop_last else int(np.ceil(len(self.y) / self.batch_size))
        skf = StratifiedKFold(n_splits=n_batches, shuffle=self.shuffle)
        X_dummy = np.zeros(len(self.y))     # dummy data, only y is used for stratification
        batches = []
        for _, test_idx in skf.split(X_dummy, self.y):
            if self.drop_last and len(test_idx) < self.batch_size:
                continue
            batches.append(test_idx)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class CustomDataset(Dataset):
    """Basic Dataset wrapper for tensors."""
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class EarlyStopping:
    """Early stops training if validation loss doesn't improve."""
    def __init__(self, patience=7, delta=0.001, path="checkpoint.pt", verbose=False, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.001
            path (str): Path for the checkpoint to be saved to.
                            Default: "checkpoint.pt"        
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping Counter: [{self.counter}/{self.patience}]")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.5f} → {val_loss:.5f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(self, model, batch_size, n_epoch, n_class, patience, loss_fn, optimizer, directory, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.n_class = n_class
        self.patience = patience
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.directory = directory

        self.calculate_f1 = MulticlassF1Score(num_classes=n_class, average="macro").to(device)
        self.tracker = {}

        for k, v in kwargs.items():
            setattr(self, k, v)

    def train(self, train_loader, val_loader):
        early_stopping = EarlyStopping(patience=self.patience, delta=0.001, path=self.directory, verbose=False)
        min_epoch = 20
        train_loss_tracker, val_loss_tracker, training_time_tracker = [], [], []

        for epoch in range(self.n_epoch):
            T0 = time.time()

            self.model.train()
            train_losses, train_f1s = [], []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                logits = self.model(data)
                loss = self.loss_fn(logits, target)
                f1 = self.calculate_f1(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                train_f1s.append(f1.item())

            val_losses, val_f1s = [], []
            self.model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    logits = self.model(data)
                    val_loss = self.loss_fn(logits, target)
                    val_f1 = self.calculate_f1(logits, target)
                    val_losses.append(val_loss.item())
                    val_f1s.append(val_f1.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_f1 = np.mean(train_f1s)
            val_f1 = np.mean(val_f1s)
            training_time = time.time() - T0

            print(f"Epoch:{epoch+1} | TrainLoss:{train_loss:.5f} | TrainF1:{train_f1:.2f} | "
                  f"ValLoss:{val_loss:.5f} | ValF1:{val_f1:.2f} | Time:{training_time:.2f}s")

            train_loss_tracker.append(train_loss)
            val_loss_tracker.append(val_loss)
            training_time_tracker.append(training_time)

            early_stopping(val_loss, self.model)
            if (epoch + 1) >= min_epoch and early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        self.tracker = {
            "train_tracker": train_loss_tracker,
            "val_tracker": val_loss_tracker,
            "training_time_tracker": training_time_tracker
        }
        return self.tracker

    def calculate_metrics(self, y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)

        f1_binary = f1_score(y_true, y_pred, average="binary")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        roc_fpr, roc_tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(roc_fpr, roc_tpr)

        p, r, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r, p)

        bacc = balanced_accuracy_score(y_true, y_pred)

        return accuracy, bacc, precision, recall, specificity, f1_binary, f1_macro, f1_weighted, roc_auc, pr_auc

    def plot_confusion_matrix(self, cm, display_labels):
        T0 = time.time()

        plt.rcParams.update({'font.size': 20})

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm_normalized, interpolation = 'nearest', cmap = plt.cm.Blues)
        # ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=display_labels, yticklabels=display_labels,
               xlabel='Predicted Label',
               ylabel='True Label')
        
        plt.setp(ax.get_xticklabels(), rotation = 0, ha='center', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation = 90, ha='center', va='bottom', rotation_mode='anchor')

        fmt = '.2f'
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm_normalized[i, j], fmt),
                        ha="center", va="center", fontsize=20,
                        color="white" if cm_normalized[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()
        fig.savefig(op.join('confusion_matrix', ''.join(display_labels)+'_'+str(T0)+'.png'))

    
    def eval_model(self, model, X_test, y_test, class_names):
        model.load_state_dict(torch.load(self.directory))
        T0 = time.time()

        test_data = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                 worker_init_fn=seed_worker, generator=g, num_workers=0)

        model.eval()
        test_losses, test_f1s, y_pred, y_prob = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)

                loss = self.loss_fn(logits, target)
                f1 = self.calculate_f1(logits, target)

                test_losses.append(loss.item())
                test_f1s.append(f1.item())

                _, preds = torch.max(logits, 1)
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(logits.softmax(dim=1).cpu().tolist())

        test_loss = np.mean(test_losses)
        test_f1 = np.mean(test_f1s)
        test_time = time.time() - T0

        y_prob = np.array(y_prob)[:, 1]  # assuming binary classification
        y_pred = np.array(y_pred)

        metrics = self.calculate_metrics(y_test, y_pred, y_prob)

        print(f"Macro F1-score: {test_f1:.4f}")

        evaluation = {
            "testing_loss": test_loss,
            "accuracy": metrics[0],
            "balanced_accuracy": metrics[1],
            "precision": metrics[2],
            "recall": metrics[3],
            "specificity": metrics[4],
            "f1_binary": metrics[5],
            "f1_macro": metrics[6],
            "f1_weighted": metrics[7],
            "roc_auc": metrics[8],
            "pr_auc": metrics[9],
            "testing_time": test_time
        }

        Y = {"y_true": y_test, "y_pred": y_pred}

        cm = confusion_matrix(y_test, y_pred, normalize='all')

        self.plot_confusion_matrix(cm, display_labels=class_names)

        return Y, evaluation

