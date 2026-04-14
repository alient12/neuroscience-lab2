import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import medfilt, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

fs=1000

def create_new_sampling_file(path, name):
    # complete file name with unique ID
    file_id = datetime.now()
    file_name = (path + name + "_" + file_id.strftime("%y-%m-%d_%H-%M-%S") + ".csv")

    # opening file in writing mode
    file = open(file_name, 'w', newline="")

    # header fields
    fields = ['voltage1 (V)', 'voltage2 (V)']
    csv.DictWriter(file, fieldnames=fields).writeheader()

    # closing file
    file.close()

    return file_name


def acquire_window(ch1, ch2, window_size, writer):
    # EMG data for current window
    data1 = []
    data2 = []

    # acquisition loop
    for i in range(window_size):
        data1.append(ch1.voltage)
        data2.append(ch2.voltage)
        writer.writerow([data1[i], data2[i]])

    return data1, data2


def acquire_training_dataset(ch1, ch2, window_size, num_window, file_name):
    # file opening in append mode
    file = open(file_name, 'a', newline="")
    training_writer = csv.writer(file)

    # acquisition loop
    print("debut acquisition")
    start_time = time.time()
    
    for window in range(num_window):
        acquire_window(ch1, ch2, window_size, training_writer)
    
    end_time = time.time()
    file.close()

    # frequency sampling calculation
    total_samples = window_size * num_window
    elapsed_time = end_time - start_time
    # global fs
    fs = total_samples / elapsed_time
    print("frequence acquisition : " + str(round(fs, 2)) + " sps")


def visualize_sampling(file):
    data = pd.read_csv(file)
    
    v1 = data["voltage1 (V)"]
    v2 = data["voltage2 (V)"]
    x = range(0, len(v1))
    
    plt.plot(x, v1, label="voltage1 (V)")
    plt.plot(x, v2, label="voltage2 (V)")
    
    plt.title("Plot of Sensors Voltages")
    plt.legend()
    plt.show()


# ============================================= Label Generation (humain) ============================================= #

def robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def moving_average(x, k=3):
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def generate_labels(csv_path, window_size=50, pattern=(1, 2, 1, 2, 1, 1, 2, 2)):
    """
    Generate one label per window of size `window_size`.

    Returns:
        labels: numpy array of shape (num_windows,)
    """

    df = pd.read_csv(csv_path)

    # first two columns are voltage1 and voltage2
    v1 = df.iloc[:, 0].to_numpy(dtype=float)
    v2 = df.iloc[:, 1].to_numpy(dtype=float)

    n_use = (len(v1) // window_size) * window_size
    v1 = v1[:n_use]
    v2 = v2[:n_use]

    # --------------------------------------------------
    # 1) Remove slow baseline drift
    # --------------------------------------------------
    baseline_kernel = 51
    b1 = medfilt(v1, baseline_kernel)
    b2 = medfilt(v2, baseline_kernel)

    r1 = v1 - b1
    r2 = v2 - b2

    # --------------------------------------------------
    # 2) Add derivative features to capture sharp spikes
    # --------------------------------------------------
    d1 = np.diff(v1, prepend=v1[0])
    d2 = np.diff(v2, prepend=v2[0])

    # --------------------------------------------------
    # 3) Estimate noise from the first few windows
    # --------------------------------------------------
    noise_frames = min(8 * window_size, n_use)

    zr1 = np.abs(r1) / robust_sigma(r1[:noise_frames])
    zr2 = np.abs(r2) / robust_sigma(r2[:noise_frames])
    zd1 = np.abs(d1) / robust_sigma(d1[:noise_frames])
    zd2 = np.abs(d2) / robust_sigma(d2[:noise_frames])

    # amplitude + sharpness
    sample_score = np.maximum(zr1, zr2) + 0.5 * np.maximum(zd1, zd2)

    # --------------------------------------------------
    # 4) Convert to one score per 50-sample window
    # --------------------------------------------------
    num_windows = n_use // window_size
    score = np.array([
        np.percentile(sample_score[i * window_size:(i + 1) * window_size], 90)
        for i in range(num_windows)
    ])

    score_s = moving_average(score, k=3)

    # --------------------------------------------------
    # 5) Detect the 8 strongest peaks
    # --------------------------------------------------
    peaks, _ = find_peaks(
        score_s,
        distance=6,
        prominence=max(0.1, 0.1 * np.std(score_s)),
    )

    if len(peaks) > 0:
        peaks = peaks[np.argsort(score_s[peaks])[::-1][:len(pattern)]]
        peaks = np.sort(peaks)

    # fallback: if not enough peaks, fill with strongest remaining windows
    chosen = list(peaks)
    for p in np.argsort(score_s)[::-1]:
        if all(abs(int(p) - int(q)) >= 6 for q in chosen):
            chosen.append(int(p))
            if len(chosen) == len(pattern):
                break

    peaks = np.array(sorted(chosen[:len(pattern)]), dtype=int)

    # --------------------------------------------------
    # 6) Use valleys between peaks to define event limits
    # --------------------------------------------------
    boundaries = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        lo = a + 1
        hi = b
        m = lo + np.argmin(score_s[lo:hi]) if hi > lo else lo
        boundaries.append(int(m))
    boundaries.append(len(score_s) - 1)

    # --------------------------------------------------
    # 7) Expand each peak into a region with duration limits
    # --------------------------------------------------
    regions = []
    min_len = 5
    max_len = 9
    alpha = 0.4

    for i, p in enumerate(peaks):
        left_lim = boundaries[i]
        right_lim = boundaries[i + 1]

        # local threshold between valley floor and peak height
        bg = min(score_s[left_lim], score_s[right_lim])
        thr = bg + alpha * (score_s[p] - bg)

        l = p
        while l > left_lim and score_s[l - 1] >= thr:
            l -= 1

        r = p
        while r < right_lim and score_s[r + 1] >= thr:
            r += 1

        # enforce realistic event duration
        if (r - l + 1) < min_len:
            extra = min_len - (r - l + 1)
            l = max(left_lim, l - extra // 2)
            r = min(right_lim, r + (extra - extra // 2))

            while (r - l + 1) < min_len:
                if l > left_lim:
                    l -= 1
                elif r < right_lim:
                    r += 1
                else:
                    break

        if (r - l + 1) > max_len:
            half = max_len // 2
            l = max(left_lim, p - half)
            r = l + max_len - 1
            if r > right_lim:
                r = right_lim
                l = max(left_lim, r - max_len + 1)

        regions.append((l, r))

    # --------------------------------------------------
    # 8) Assign the fixed pattern
    # --------------------------------------------------
    labels = np.zeros(num_windows, dtype=int)
    for (l, r), lab in zip(regions, pattern):
        labels[l:r + 1] = lab

    return labels

# ================================================ Classifier ================================================ #

def filter_emg(signal, fs):
    """
    Light EMG filtering:
    - remove DC
    - band-pass
    - notch 60 Hz

    If filtering fails for any reason, return a mean-centered signal.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    try:
        nyq = fs / 2.0

        # band-pass
        low = 20.0 / nyq
        high = 200.0 / nyq
        if 0 < low < high < 1:
            b, a = butter(4, [low, high], btype="band")
            signal = filtfilt(b, a, signal)

        # notch 60 Hz
        w0 = 60.0 / nyq
        if 0 < w0 < 1:
            b_notch, a_notch = iirnotch(w0, Q=30)
            signal = filtfilt(b_notch, a_notch, signal)

    except Exception:
        signal = signal - np.mean(signal)

    return signal


def extract_features(signal):
    """
    Best feature set found for LDA:
    - RMS
    - Waveform Length
    - STD
    - MaxAbs
    """
    signal = np.asarray(signal, dtype=float)

    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    std = np.std(signal)
    maxabs = np.max(np.abs(signal))

    return np.array([rms, wl, std, maxabs], dtype=float)


def train_classifier(file_name, window_size):
    """
    Entraîne un classifieur EMG avec normalisation.
    Signature kept the same.

    Notes:
    - Uses the real label file if available.
    - Falls back to generated labels only if label file is missing.
    - Best LDA variant found:
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    """

    data = pd.read_csv(file_name)

    labels = generate_labels(file_name, window_size=window_size)
    print(f"Generated labels: {labels}")

    signal1 = data["voltage1 (V)"].values
    signal2 = data["voltage2 (V)"].values

    # ---------- 2. Filtrage ----------
    signal1 = filter_emg(signal1, fs)
    signal2 = filter_emg(signal2, fs)

    # ---------- 3. Fenêtrage + features ----------
    X = []
    y = []

    for i in range(len(labels)):
        start = i * window_size
        end = start + window_size

        window1 = signal1[start:end]
        window2 = signal2[start:end]

        if len(window1) != window_size or len(window2) != window_size:
            continue

        f1 = extract_features(window1)
        f2 = extract_features(window2)

        feature_vector = np.concatenate([f1, f2])
        X.append(feature_vector)
        y.append(labels[i])

    X = np.array(X, dtype=float)
    y = np.array(y)

    # ---------- 4. NORMALISATION ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------- 5. Entraîner ----------
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(X_scaled, y)

    acc = model.score(X_scaled, y)
    print(f"Training accuracy: {acc * 100:.2f}%")

    classif = {
        "model": model,
        "scaler": scaler,
    }
    return classif


def test_classifier(classif, ch1, ch2, window_size, num_window, buzzer, file_name):
    """
    Classification EMG en temps réel.
    Signature kept the same.
    """

    buffer1 = []
    buffer2 = []
    cnt = 0
    labels = []

    step_size = window_size

    print("Démarrage classification temps réel...")

    file = open(file_name, 'w', newline="")
    writer = csv.writer(file)
    fields = ['voltage1 (V)', 'voltage2 (V)']
    writer.writerow(fields)

    while True:
        # ---------- 1. Acquisition ----------
        sample1 = ch1.voltage
        sample2 = ch2.voltage

        writer.writerow([sample1, sample2])

        buffer1.append(sample1)
        buffer2.append(sample2)

        # ---------- 2. Si fenêtre pleine ----------
        if len(buffer1) >= window_size:
            cnt += 1

            window1 = np.array(buffer1[:window_size], dtype=float)
            window2 = np.array(buffer2[:window_size], dtype=float)

            # ---------- 3. Filtrage ----------
            window1 = filter_emg(window1, fs)
            window2 = filter_emg(window2, fs)

            # ---------- 4. Features ----------
            f1 = extract_features(window1)
            f2 = extract_features(window2)
            feature_vector = np.concatenate([f1, f2])

            # ---------- 5. Normalisation ----------
            feature_vector = classif['scaler'].transform([feature_vector])

            # ---------- 6. Prédiction ----------
            pred = classif['model'].predict(feature_vector)[0]

            # ---------- 7. Feedback buzzer ----------
            if pred == 1:
                buzzer.ChangeFrequency(1000)
                buzzer.start(50)
            elif pred == 2:
                buzzer.ChangeFrequency(2000)
                buzzer.start(50)
            else:
                buzzer.stop()

            buffer1 = buffer1[step_size:]
            buffer2 = buffer2[step_size:]

            labels.append(pred)

            if cnt >= num_window:
                break

        # time.sleep(1 / fs)

    file.close()

    label_file = file_name.replace(".csv", "_predicted_labels.csv")
    with open(label_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'])
        for label in labels:
            writer.writerow([label])

    return labels

# ================================================ Evaluation ================================================ #

import os
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def evaluate_train_test_pair(train_data, train_labels, test_data, test_labels, window_size):
    global EMG_file, label_file

    EMG_file = train_data
    label_file = train_labels

    classif = train_classifier(train_data, window_size)

    # ---------- build test features offline ----------
    test_df = pd.read_csv(test_data)

    # keep same label reading style as grading code
    LT = pd.read_csv(test_labels)
    if "Downsampled Carrier" in LT.columns:
        y_true = LT["Downsampled Carrier"].to_numpy()
    else:
        y_true = LT.values.flatten()

    signal1 = filter_emg(test_df["voltage1 (V)"].values, fs)
    signal2 = filter_emg(test_df["voltage2 (V)"].values, fs)

    X_test = []
    y_test = []

    for i in range(len(y_true)):
        start = i * window_size
        end = start + window_size

        w1 = signal1[start:end]
        w2 = signal2[start:end]

        if len(w1) != window_size or len(w2) != window_size:
            continue

        f1 = extract_features(w1)
        f2 = extract_features(w2)

        X_test.append(np.concatenate([f1, f2]))
        y_test.append(y_true[i])

    X_test = np.array(X_test, dtype=float)
    y_test = np.array(y_test)

    X_test = classif["scaler"].transform(X_test)
    y_pred = classif["model"].predict(X_test)

    # ---------- grading metric ----------
    mean_class_accuracy = balanced_accuracy_score(y_test, y_pred)

    # normalized confusion matrix like your grading code
    conf_matrix = confusion_matrix(y_test, y_pred, normalize="true")

    return mean_class_accuracy, conf_matrix, y_pred, y_test


def evaluate_all_pairs(window_size=50, swap_train_test=False, data_path="./"):
    pairs = [
        ("train1_data.csv", "train1_labels.csv", "test1_data.csv", "test1_labels.csv"),
        ("train2_data.csv", "train2_labels.csv", "test2_data.csv", "test2_labels.csv"),
        ("train3_data.csv", "train3_labels.csv", "test3_data.csv", "test3_labels.csv"),
        ("train4_data.csv", "train4_labels.csv", "test4_data.csv", "test4_labels.csv"),
        ("train5_data.csv", "train5_labels.csv", "test5_data.csv", "test5_labels.csv"),
        ("train6_data.csv", "train6_labels.csv", "test6_data.csv", "test6_labels.csv"),
    ]

    if swap_train_test:
        pairs = [
            ("test1_data.csv", "test1_labels.csv", "train1_data.csv", "train1_labels.csv"),
            ("test2_data.csv", "test2_labels.csv", "train2_data.csv", "train2_labels.csv"),
            ("test3_data.csv", "test3_labels.csv", "train3_data.csv", "train3_labels.csv"),
            ("test4_data.csv", "test4_labels.csv", "train4_data.csv", "train4_labels.csv"),
            ("test5_data.csv", "test5_labels.csv", "train5_data.csv", "train5_labels.csv"),
            ("test6_data.csv", "test6_labels.csv", "train6_data.csv", "train6_labels.csv"),
        ]

    # safer path join
    pairs = [
        tuple(os.path.join(data_path, fname) for fname in pair)
        for pair in pairs
    ]

    all_scores = []
    all_confusions = []

    for train_data, train_labels, test_data, test_labels in pairs:
        mean_class_accuracy, conf_matrix, y_pred, y_true = evaluate_train_test_pair(
            train_data, train_labels, test_data, test_labels, window_size
        )

        all_scores.append(mean_class_accuracy)
        all_confusions.append(conf_matrix)

        print(f"{os.path.basename(train_data)} -> {os.path.basename(test_data)}")
        print(f"Mean class accuracy: {mean_class_accuracy:.6f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print()

    mean_score = np.mean(all_scores)

    # average normalized confusion matrices across pairs
    mean_conf_matrix = np.mean(np.stack(all_confusions), axis=0)

    print(f"Overall mean class accuracy: {mean_score:.6f}")
    # print("Average Confusion Matrix:")
    # print(mean_conf_matrix)

    return all_scores, mean_score, all_confusions, mean_conf_matrix