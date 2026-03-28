import os
import csv
import mne
import glob
import numpy as np

from tqdm import tqdm

def write_log(filepath, data, mode="w"):
    """Write a single row to a CSV log file."""
    try:
        with open(filepath, mode) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)
    except IOError:
        raise IOError("Unable to write log file")
    
def save_data_train_val_test(save_dir, subject, fold, 
                             X_train, y_train, 
                             X_val, y_val, 
                             X_test, y_test):
    """
    Save training, validation, and test datasets to .npy files with subject/fold identifiers.

    Parameters
    ----------
    save_dir : str
        Directory where the files will be saved.
    subject : int
        Subject index (1-based in filenames).
    fold : int
        Fold index (1-based in filenames).
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Data arrays to save.

    Raises
    ------
    IOError
        If saving any of the files fails.
    """
    file_name = f"S{subject:02d}_f{fold:02d}"

    try:
        np.save(f"{save_dir}/X_train_{file_name}.npy", X_train)
        np.save(f"{save_dir}/y_train_{file_name}.npy", y_train)

        np.save(f"{save_dir}/X_val_{file_name}.npy", X_val)
        np.save(f"{save_dir}/y_val_{file_name}.npy", y_val)

        np.save(f"{save_dir}/X_test_{file_name}.npy", X_test)
        np.save(f"{save_dir}/y_test_{file_name}.npy", y_test)

    except Exception as e:
        raise IOError(f"Failed to save dataset files for {file_name} in {save_dir}: {e}")
    
def load_data_train_val_test(save_dir, subject, fold):
    """
    Load training, validation, and test datasets from .npy files.

    Parameters
    ----------
    save_dir : str
        Directory where the dataset files are stored.
    subject : int
        Subject index (1-based in filenames).
    fold : int
        Fold index (1-based in filenames).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Loaded datasets.

    Raises
    ------
    IOError
        If any of the required files cannot be found or loaded.
    """
    file_name = f"S{subject:02d}_f{fold:02d}"

    try:
        X_train = np.load(f"{save_dir}/X_train_{file_name}.npy")
        y_train = np.load(f"{save_dir}/y_train_{file_name}.npy")

        X_val = np.load(f"{save_dir}/X_val_{file_name}.npy")
        y_val = np.load(f"{save_dir}/y_val_{file_name}.npy")

        X_test = np.load(f"{save_dir}/X_test_{file_name}.npy")
        y_test = np.load(f"{save_dir}/y_test_{file_name}.npy")

    except FileNotFoundError as e:
        raise IOError(f"Missing dataset file for {file_name} in {save_dir}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load dataset files for {file_name} in {save_dir}: {e}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data(experiment, task, window_duration, onset, dataset_path):
    """
    Load the preprocessed EEG epochs for Motor Execution (ME) or Motor Imagery (MI) experiments.

    This function reads preprocessed EEG files (in `.fif` format) from the specified dataset directory,
    extracts task-related and rest-related windows, and generates input arrays with corresponding labels.
    It supports both ME and MI experiments with flexible task definitions and window durations.

    Parameters
    ----------
    experiment : str
        Experiment type, either "ME" (motor execution) or "MI" (motor imagery).
    task : str
        Task name. Either "sit_std" or "std_sit".
    window_duration : float
        Length of the EEG time window (in seconds) to extract from each trial.
    onset : float
        Time offset (in seconds) from event onset to define the extraction window.
        For ME, the window ends at `onset`; for MI, the window starts at `onset`.
    dataset_path : str
        Base directory where the processed EEG dataset is stored. Data should be under
        `<dataset_path>/<EXPERIMENT>/*.fif`.

    Returns
    -------
    X : list of ndarray
        List of EEG trial arrays of shape (n_epochs, n_channels, n_times).
        Each element contains concatenated rest and task segments.
    y : list of ndarray
        List of label arrays corresponding to `X`, where 0 = rest and 1 = task.

    Notes
    -----
    - Trials with mismatched or incomplete window lengths are skipped.
    - Rest epochs are matched to task epochs by event indices to maintain alignment.
    """

    tasks = ["sit_std", "std_sit"]
    if experiment.upper() == "ME":
        target_label = f"{experiment.lower()}_{task}"
        another_label = f"{experiment.lower()}_{tasks[1]}" if tasks[0] == task else f"{experiment.lower()}_{tasks[0]}"
        rest_label = "me_r"
    elif experiment.upper() == "MI":
        initial_pos = task.split("_")[0]
        end_pos = task.split("_")[1]
        target_label = f"{experiment.lower()}_{initial_pos}_{end_pos}"
        another_label = f"{experiment.lower()}_{initial_pos}_{initial_pos}"
        rest_label = f"mi_r_{initial_pos}"

    raw_path = os.path.join(dataset_path, experiment.upper(), "*.fif")
    raw_files = sorted(glob.glob(raw_path))

    X, y = [], []
    for file in tqdm(raw_files):
        epochs = mne.read_epochs(file, verbose=False)
        sfreq = int(epochs.info["sfreq"])
        event_id = epochs.event_id

        # Get data for each class
        target_data = epochs[target_label].get_data()
        rest_data = epochs[rest_label].get_data()

        # Match rest epochs using indices of target_label/another_label
        indices = np.where(epochs[[target_label, another_label]].events[:, 2] == event_id[target_label])[0]
        rest_data = rest_data[indices[indices < len(rest_data)]]

        onset_timepoint = int(sfreq * onset)
        window_length = int(sfreq * window_duration)

        # Extract appropriate window
        if experiment.upper() == "ME":
            start_idx = max(0, onset_timepoint - window_length)
            rest_segments = rest_data[:, :, start_idx:onset_timepoint]
            target_segments = target_data[:, :, start_idx:onset_timepoint]        
        elif experiment.upper() == "MI":
            rest_segments = rest_data[:, :, onset_timepoint:onset_timepoint + window_length]
            target_segments = target_data[:, :, onset_timepoint:onset_timepoint + window_length]

        # Skip if bad shape
        if rest_segments.shape[-1] != window_length or target_segments.shape[-1] != window_length:
            continue

        # Labels: 0 = rest, 1 = task
        rest_labels = np.zeros(len(rest_segments), dtype=int)
        target_labels = np.ones(len(target_segments), dtype=int)

        X.append(np.concatenate([rest_segments, target_segments]))
        y.append(np.concatenate([rest_labels, target_labels]))
    
    class_name = [rest_label, target_label]
    return X, y, class_name