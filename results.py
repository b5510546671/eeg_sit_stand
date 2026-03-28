from __future__ import annotations
import pandas as pd
import os
import warnings
import numpy as np

import pingouin as pg
import os.path as op
from glob import glob
from pathlib import Path
from tqdm.notebook import tqdm

# Visualization settings
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")
warnings.filterwarnings("ignore")


# Column configurations
head_cols = ["model_name", "experiment", "task", "duration", "subject", "fold"]
metric_cols = ["accuracy", "f1_binary", "roc_auc", "training_time", "testing_time"]


def create_experiment_dataframe(log_path: str) -> pd.DataFrame:
    def _parse_metadata_from_path(path: Path):
        try:
            model_name, experiment, task, duration, filename = path.parts[-5:]
            model_name = model_name.replace("_results","")
            # model_name = model_name.removesuffix("_results")
            prefix = f"report_{model_name}_S"
            if not (filename.startswith(prefix) and filename.endswith(".csv")):
                return None
            subject = int(filename[len(prefix):-4])
            return model_name, experiment, task, duration, subject
        except Exception:
            return None

    def _load_loss_training_times(folder: Path, model_name: str, subject: int):
        out = {}
        for npz in folder.glob(f"loss_{model_name}_S{subject:02d}_f*.npz"):
            try:
                fold = int(npz.stem.rsplit("_f", 1)[-1])
                loss = np.load(npz)
                if "training_time_tracker" in loss:
                    out[fold] = float(np.sum(loss["training_time_tracker"]))
                else:
                    warnings.warn(f"'training_time_tracker' missing in {npz}")
            except Exception as e:
                warnings.warn(f"Skip NPZ {npz}: {e}")
        return out

    root_path = Path(log_path)
    if not root_path.exists():
        raise FileNotFoundError(f"log_path not found: {log_path}")

    frames = []
    for dpath, dnames, fnames in os.walk(root_path):
        if dnames:  # skip non-leaves
            continue
        folder = Path(dpath)
        for fname in (f for f in fnames if f.endswith(".csv") and f.startswith("report_")):
            csv_path = folder / fname
            meta = _parse_metadata_from_path(csv_path)
            if meta is None:
                warnings.warn(f"Skip (path pattern mismatch): {csv_path}")
                continue
            model_name, experiment, task, duration, subject = meta
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                warnings.warn(f"Cannot read CSV {csv_path}: {e}")
                continue

            # Ensure 'fold'
            if "fold" not in df.columns:
                df = df.copy()
                df.insert(0, "fold", np.arange(len(df), dtype=int))
                warnings.warn(f"'fold' missing in {csv_path}; created 0..N-1.")

            # Insert metadata
            df = df.copy()
            df.insert(0, "model_name", model_name)
            df.insert(1, "experiment", experiment)
            df.insert(2, "task", task)
            df.insert(3, "duration", duration)
            df.insert(4, "subject", subject)

            # Compute training_time and attach (will be kept via metric_cols)
            fold2time = _load_loss_training_times(folder, model_name, subject)
            df["training_time"] = df["fold"].map(fold2time).astype(float)

            # Keep requested columns that exist
            keep = [c for c in (head_cols + metric_cols) if c in df.columns]
            missing = [c for c in metric_cols if c not in df.columns]
            if missing:
                warnings.warn(f"Missing metrics {missing} in {csv_path}")
            df = df[keep]

            # Cast metrics to float (including training_time)
            for c in metric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            frames.append(df)
            
    if not frames:
        return pd.DataFrame(columns=head_cols + metric_cols)

    out = pd.concat(frames, ignore_index=True)
    # Final coercion (no-op if done above, keeps robustness)
    for c in metric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out



def create_experiment_summary(all_records: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a formatted summary DataFrame that includes mean ± std for each metric,
    highlights the best value with Parentheses `( )`, and marks statistically significant
    differences with an asterisk `*`.
    """
    subject_scores = all_records.groupby([x for x in head_cols if x not in ["fold"]]).mean()
    grouped = subject_scores.groupby([x for x in head_cols if x not in ["fold", "subject"]])
    scores = grouped.mean()

    for metric in metric_cols:
        m = grouped[metric].mean()
        s = grouped[metric].std()

        if metric == "roc_auc":
            scores[metric] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(m, s)]
        elif metric in ["training_time", "testing_time"]:
            scores[metric] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(m, s)]
        else:
            scores[metric] = [f"{m*100:.2f} ± {s*100:.2f}" for m, s in zip(m, s)]

        for (experiment, task, model_name), df in m.groupby(level=[0, 1, 2]):
            if metric in ["training_time", "testing_time"]:
                best_idx = df.idxmin()
            else:
                best_idx = df.idxmax()

            scores.loc[best_idx, metric] = f"({scores.loc[best_idx, metric]})"

            dff = subject_scores.loc[(experiment, task, model_name)].reset_index()
            is_normal = pg.normality(data=dff, method="shapiro", alpha=0.05).loc[metric, "normal"]
            if is_normal:
                p_val = pg.rm_anova(data=dff, dv=metric, within="duration", subject="subject").loc[0, "p-unc"]
            else:
                p_val = pg.friedman(data=dff, dv=metric, within="duration", subject="subject").loc["Friedman", "p-unc"]

            sig_pair = []
            if p_val <= 0.05:
                pairwise = pg.pairwise_tests(data=dff, dv=metric, within="duration", subject="subject", parametric=is_normal, alpha=0.05)
                pairwise = pairwise[(pairwise["A"] == best_idx[-1]) | (pairwise["B"] == best_idx[-1])]
                sig_ttest = pairwise[pairwise["p-unc"] <= 0.05]
                sig_pair = [x for x in sig_ttest[["A", "B"]].values.flatten() if x != best_idx[-1]]

            for duration in sig_pair:
                sig_idx = (experiment, task, model_name, duration)
                if sig_idx in scores.index:
                    scores.loc[sig_idx, metric] += "*"

    # Drop unneeded column if present
    scores = scores.drop(columns=["fold"], errors="ignore")

    return scores





# Run and preview
all_records = create_experiment_dataframe(log_path=op.join("log_raw", "TCANet_results"))
# all_records = create_experiment_dataframe(log_path=op.join("log_raw"))



summary = create_experiment_summary(all_records)
print(summary)



import matplotlib.pyplot as plt

# --- group by subject ---
df_grouped = all_records.groupby("subject").mean(numeric_only=True).reset_index()

# --- sort by accuracy ---
df_grouped = df_grouped.sort_values(by="accuracy").reset_index(drop=True)
df_grouped["sorted_subject"] = df_grouped.index + 1  # make 1..N index

# --- plot ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Accuracy line
ax1.plot(df_grouped["sorted_subject"], df_grouped["accuracy"],
         color="tab:blue", marker="o", label="Accuracy")
ax1.set_xlabel("Sorted Subject (by accuracy)")
ax1.set_ylabel("Accuracy (%)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Training time line (secondary y-axis)
ax2 = ax1.twinx()
ax2.plot(df_grouped["sorted_subject"], df_grouped["training_time"],
         color="tab:red", marker="s", label="Training Time")
ax2.set_ylabel("Training Time (s)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Grid + title
fig.suptitle("Accuracy & Training Time (Subjects Sorted by Accuracy)", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.6)

plt.show()


print(df_grouped)

summary.to_csv('summary.csv')



# all_records['duration'] = all_records['duration'].str.extract(r'(\d+\.\d+)').astype(float)
all_records['accuracy'] *= 100
all_records['f1_binary'] *= 100
all_records['roc_auc'] *= 100




import matplotlib.pyplot as plt
import seaborn as sns

x = 'subject'
x_new = 'sorted_subject'
y = 'accuracy'
hue = 'duration'
duration_to_sort = '2.0s'

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

for ax, (exp, task) in zip(axes.ravel(), [('ME', 'sit_std'), ('ME', 'std_sit'), ('MI', 'sit_std'), ('MI', 'std_sit')]):
    # --- ME experiment ---
    dff = all_records[(all_records['experiment']==exp) & (all_records['task']==task)]
    hue_order = sorted(dff[hue].unique())
    dff = dff.groupby(['model_name', 'experiment', 'task', hue, x]).mean().reset_index()
    mapping_dict = dff[dff[hue]==duration_to_sort].sort_values(by=y).reset_index(drop=True).rename_axis(x_new).reset_index()[[x_new, x]].sort_values(by=x).set_index(x).to_dict()[x_new]
    dff[x_new] = dff[x].map(mapping_dict)
    sns.lineplot(ax=ax, 
                data=dff, 
                x=x_new, y=y, hue=hue, 
                linestyle='-', marker="o",
                palette=sns.color_palette("viridis", len(hue_order)), 
                hue_order=hue_order,
                errorbar=None,
                )

    ax.set_title("{} task in {} experiment".format(task.upper(), exp.upper()))
    ax.set_xlabel(x.capitalize())
    ax.set_ylabel(y.capitalize())
    ax.set_ylim(45, 100)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, ncol=1, title="Duration", loc='upper left', edgecolor='white')

plt.tight_layout()
plt.savefig("result_plot.pdf", dpi=300, bbox_inches='tight')
plt.show()
