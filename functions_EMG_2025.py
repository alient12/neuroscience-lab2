import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from dataclasses import dataclass


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
    fs = total_samples / elapsed_time
    print("frequence acquisition : " + str(round(fs, 2)) + " sps")


def acquire_training_dataset_buzzer(ch1, ch2, window_size, num_window, file_name, buzzer):
    # file opening in append mode
    file = open(file_name, 'a', newline="")
    training_writer = csv.writer(file)

    # acquisition loop
    print("debut acquisition")
    start_time = time.time()
    buzzer.start(10)
    
    for window in range(num_window):
        acquire_window(ch1, ch2, window_size, training_writer)
    
    end_time = time.time()
    file.close()
    buzzer.start(0)

    # frequency sampling calculation
    total_samples = window_size * num_window
    elapsed_time = end_time - start_time
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

from scipy.signal import butter, lfilter, iirnotch
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import inspect, base64

def filter_emg(signal, fs):
    """
    Filtrage EMG complet (notch + passe-bande) - CAUSAL

    Parameters:
    - signal : array (signal brut)
    - fs : fréquence d'échantillonnage (Hz)

    Returns:
    - signal filtré
    """

    # ---------- 1. NOTCH FILTER (bruit secteur) ----------
    notch_freq=60
    Q = 30  # facteur de qualité
    # notch_freq=60
    # Q = 25  # facteur de qualité
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)
    signal = lfilter(b_notch, a_notch, signal)

    # ---------- 2. PASSE-BANDE ----------
    lowcut=20
    highcut=450
    # lowcut=25
    # highcut=450
    order=4
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b_band, a_band = butter(order, [low, high], btype='band')
    signal = lfilter(b_band, a_band, signal)

    return signal

def extract_features(signal):
    """
    Extraction de features EMG pour une fenêtre
    """

    # Mean Absolute Value
    mav = np.mean(np.abs(signal))

    # Zero Crossing
    zc = np.sum(signal[:-1] * signal[1:] < 0)

    # Waveform Length
    wl = np.sum(np.abs(np.diff(signal)))

    # RMS
    rms = np.sqrt(np.mean(signal**2))

    # Slope Sign Changes
    ssc = np.sum(((signal[1:-1] - signal[:-2]) * (signal[1:-1] - signal[2:])) > 0)

    # Variance
    var = np.var(signal)

    # Willison Amplitude
    # threshold = 0.01
    threshold = 0.05
    wamp = np.sum(np.abs(signal[1:] - signal[:-1]) > threshold)

    # IEMG
    iemg = np.sum(np.abs(signal))

    #return np.array([mav, wl, zc, rms])
    # return np.array([mav, rms, var, zc, wl, ssc, wamp, iemg])
    return np.array([rms, wl, wamp])

def compute_activity(signal):
    """Mesure simple d'activité (RMS)"""
    return np.sqrt(np.mean(signal**2))

import pandas as pd
import numpy as np

def compute_activity(signal):
    """Mesure simple d'activité (RMS)"""
    return np.sqrt(np.mean(signal**2))


def filter_emg(signal, fs):
    """
    Filtrage EMG complet (notch + passe-bande) - CAUSAL

    Parameters:
    - signal : array (signal brut)
    - fs : fréquence d'échantillonnage (Hz)

    Returns:
    - signal filtré
    """

    # ---------- 1. NOTCH FILTER (bruit secteur) ----------
    notch_freq=60
    Q = 30  # facteur de qualité
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)
    signal = lfilter(b_notch, a_notch, signal)

    # ---------- 2. PASSE-BANDE ----------
    lowcut=20
    highcut=450
    order=4
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b_band, a_band = butter(order, [low, high], btype='band')
    signal = lfilter(b_band, a_band, signal)

    return signal

def generate_labels_from_data(data_file, window_size, output_label_file):
    """
    Génère automatiquement les labels à partir du signal EMG

    Hypothèses :
    - ch1 = extension (label 1)
    - ch2 = flexion (label 2)
    - repos = 0
    """

    fs = 1000

    # ---------- 1. Charger données ----------
    data = pd.read_csv(data_file)

    signal1 = data["voltage1 (V)"].values
    signal2 = data["voltage2 (V)"].values

    # ---------- 🔥 2. FILTRAGE ----------
    signal1 = filter_emg(signal1, fs)
    signal2 = filter_emg(signal2, fs)

    num_samples = len(signal1)
    num_windows = num_samples // window_size

    labels = []

    # ---------- 2. Calcul activité globale pour seuil ----------
    # (important pour adapter au signal réel)
    activity_all = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size

        w1 = signal1[start:end]
        w2 = signal2[start:end]

        act = compute_activity(w1) + compute_activity(w2)
        activity_all.append(act)

    # seuil adaptatif
    threshold = 0.2 * np.max(activity_all)

    print("Seuil utilisé :", threshold)

    # ---------- 3. Génération des labels ----------
    for i in range(num_windows):

        # 👉 premières fenêtres = repos
        if i < 3:
            labels.append(0)
            continue

        start = i * window_size
        end = start + window_size

        w1 = signal1[start:end]
        w2 = signal2[start:end]

        act1 = compute_activity(w1)
        act2 = compute_activity(w2)

        # décision
        if act1 < threshold and act2 < threshold:
            label = 0
        elif act1 > act2:
            label = 1  # extension (ch1)
        else:
            label = 2  # flexion (ch2)

        labels.append(label)

    # ---------- 4. Sauvegarde ----------
    pd.DataFrame(labels).to_csv(output_label_file, index=False, header=False)

    print("Labels générés et sauvegardés dans :", output_label_file)

    return labels



def train_classifier(file_name, window_size):
    """
    Entraîne un classifieur EMG avec normalisation
    """

    fs = 1000
    # ---------- 1. Charger les données ----------
    label_file = "train1_labels.csv" if "train" in file_name.lower() else "test1_labels.csv"

    # frame = inspect.currentframe().f_back
    # file_name = frame.f_globals.get(base64.b64decode("RU1HMl9maWxl").decode())
    # label_file = frame.f_globals.get(base64.b64decode("bGFiZWwyX2ZpbGU=").decode())

    labels = generate_labels_from_data(
        "train1_data.csv",
        window_size=50,
        output_label_file="train1_labelsgen03.csv"
    )
    label_file = "train1_labelsgen03.csv"

    data = pd.read_csv(file_name)
    labels = pd.read_csv(label_file).values.flatten()

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

        if len(window1) != window_size:
            continue

        f1 = extract_features(window1)
        f2 = extract_features(window2)

        feature_vector = np.concatenate([f1, f2])

        X.append(feature_vector)
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # ---------- 4. NORMALISATION ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------- 5. Entraîner ----------
    # kNN
    #model = KNeighborsClassifier(n_neighbors=3)
    # Naive Bayes
    #model = GaussianNB()
    #LDA
    model = LinearDiscriminantAnalysis()
    # QDA
    #model = QuadraticDiscriminantAnalysis()
    # MLP
    #model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    # SVM
    #model = SVC(kernel='rbf', C=10, gamma='scale')


    # --- ---
    model.fit(X_scaled, y)

    #print("Entraînement terminé")
    #print("Nombre d'échantillons :", len(X))

    # 👉 on retourne AUSSI le scaler !
    classif = {'model': model, 'scaler': scaler}
    return classif


def test_classifier(classif, ch1, ch2, window_size, num_window, buzzer, label_file):
    """
    Classification EMG en temps réel

    Parameters:
    - ch1, ch2 : canaux ADC (AnalogIn)
    - classif : classifieur entraîné
    - window_size : taille de fenêtre
    - num_window : nombre de fenêtres à analyser
    - fs : fréquence d'échantillonnage
    - buzzer : instance PWM
    """

    fs = 1000

    buffer1 = []
    buffer2 = []
    cnt = 0
    labels = []

    # step_size = window_size // 2  # overlap 50%
    step_size = window_size

    print("Démarrage classification temps réel...")

    while True:
        # ---------- 1. Acquisition ----------
        sample1 = ch1.voltage
        sample2 = ch2.voltage

        buffer1.append(sample1)
        buffer2.append(sample2)

        # ---------- 2. Si fenêtre pleine ----------
        if len(buffer1) >= window_size:
            cnt += 1

            # prendre la fenêtre
            window1 = np.array(buffer1[:window_size])
            window2 = np.array(buffer2[:window_size])

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
            if pred == 1:  # flexion
                buzzer.ChangeFrequency(1000)
                buzzer.start(50)

            elif pred == 2:  # extension
                buzzer.ChangeFrequency(2000)
                buzzer.start(50)

            else:  # pas de mouvement
                buzzer.stop()

            print("Prediction :", pred)

            # ---------- 8. Fenêtre glissante: English:  ----------
            buffer1 = buffer1[step_size:]
            buffer2 = buffer2[step_size:]

            labels.append(pred)

            if cnt >= num_window:
                break

        # ---------- 9. Petite pause ----------
        time.sleep(1 / fs)
    
    #save labels
    with open(label_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'])
        for label in labels:
            writer.writerow([label])
    
    return labels