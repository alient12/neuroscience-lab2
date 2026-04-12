import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from dataclasses import dataclass
from scipy.signal import butter, lfilter, iirnotch
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import inspect, base64

fs=240
ONLINE = False
label_threshold = 0.1
sequence = [1, 2, 1, 2, 1, 1, 2, 2]

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
    global fs
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


def filter_emg(signal, fs):
    """
    Filtrage EMG complet (notch + passe-bande) - CAUSAL

    Parameters:
    - signal : array (signal brut)
    - fs : fréquence d'échantillonnage (Hz)

    Returns:
    - signal filtré
    """

    # ---------- 2. PASSE-BANDE ----------
    lowcut=20
    highcut=100
    order=4
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b_band, a_band = butter(order, [low, high], btype='band')
    signal = lfilter(b_band, a_band, signal)

    return signal


# ============================================= Label Generation (humain) ============================================= #



# Calculate the list of 0 with minimum 0

def min_zero_sequences(lst):
    lengths = set()
    count = 0

    for x in lst:
        if x == 0:
            count += 1
        else:
            if count >= 3:
                lengths.add(count)
            count = 0

    # gérer la fin
    if count >= 3:
        lengths.add(count)

    return min(lengths)


# Processing of the signal

def remove_isolated_movements(labels, zero_padding=2):
    cleaned = labels.copy()
    n = len(labels)

    for i in range(n):
        if labels[i] != 0:
            # vérifier zéros à gauche
            left_ok = all(
                i - j >= 0 and labels[i - j] == 0
                for j in range(1, zero_padding + 1)
            )

            # vérifier zéros à droite
            right_ok = all(
                i + j < n and labels[i + j] == 0
                for j in range(1, zero_padding + 1)
            )

            if left_ok and right_ok:
                cleaned[i] = 0  # bruit → supprimé

    return cleaned

def segment_signal(labels, liste_0):
    segments = []
    current = []
    zero_count = 0

    for x in labels:
        current.append(x)

        if x == 0:
            zero_count += 1
        else:
            if zero_count >= liste_0:
                split_index = len(current) - (zero_count + 1)

                if split_index > 0:
                    segments.append(current[:split_index])

                segments.append([0] * zero_count)
                current = [x]

            zero_count = 0

    # 🔥 CORRECTION FIN DE SIGNAL
    if current:
        if zero_count >= liste_0:
            split_index = len(current) - zero_count

            if split_index > 0:
                segments.append(current[:split_index])

            segments.append([0] * zero_count)
        else:
            segments.append(current)

    return segments



def correct_segments(segments, sequence):
    corrected_segments = []
    movement_idx = 0

    for seg in segments:
        # détecter si c'est un mouvement (pas que des 0)
        if all(x == 0 for x in seg):
            corrected_segments.append(seg)
        else:
            if movement_idx < len(sequence):
                target = sequence[movement_idx]
                movement_idx += 1
            else:
                target = seg[0]  # fallback

            corrected_segments.append([target] * len(seg))

    return corrected_segments

def reconstruct_signal(segments):
    result = []
    for seg in segments:
        result.extend(seg)
    return result


def process_signal(labels, liste_0, sequence):

    labels_clean = remove_isolated_movements(labels, zero_padding=2)
    segments = segment_signal(labels_clean, liste_0)
    corrected_segments = correct_segments(segments, sequence)
    final_signal = reconstruct_signal(corrected_segments)

    return final_signal, segments, corrected_segments


def generate_labels_from_data(data_file, window_size, output_label_file, threshold):
    """
    Génère automatiquement les labels à partir du signal EMG

    Hypothèses :
    - ch1 = extension (label 1)
    - ch2 = flexion (label 2)
    - repos = 0
    """

    # ---------- 1. Charger données ----------
    data = pd.read_csv(data_file)

    signal1 = data["voltage1 (V)"].values
    signal2 = data["voltage2 (V)"].values

    # ---------- 🔥 2. FILTRAGE ----------
    signal1 = filter_emg(signal1, fs)
    signal2 = filter_emg(signal2, fs)

    signal = signal1 + signal2

    num_samples = len(signal)
    num_windows = num_samples // window_size

    labels = []

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

        w = w1 + w2

        max_w = np.max(w)

        # décision
        if max_w > threshold:
            label = 4
        else:
            label = 0
        labels.append(label)

    liste_0 = min_zero_sequences(labels)
    final_labels, segments, corrected_segments = process_signal(labels, liste_0, sequence)

    # ---------- 4. Sauvegarde ----------

    print(final_labels)
    #print(labels)
    #print(segments)
    #print("min 0 = ", liste_0)

    pd.DataFrame(final_labels).to_csv(output_label_file, index=False, header=False)

    print("Labels générés et sauvegardés dans :", output_label_file)

    return final_labels


def visualize_sampling_filter(file, threshold, fs):
    data = pd.read_csv(file)

    v1 = data["voltage1 (V)"].values
    v2 = data["voltage2 (V)"].values

    v1 = filter_emg(v1, fs)
    v2 = filter_emg(v2, fs)

    v = v1 + v2

    x = range(0, len(v))

    plt.plot(x, v, label="voltage (V) (filtré)", color='orange')

    # threshold
    plt.axhline(y=threshold, linestyle='--', label="threshold")

    plt.title("Plot of Sensors Voltages (filtrés)")
    plt.legend()
    plt.show()

# ================================================ Classifier ================================================ #


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


def train_classifier(file_name, window_size):
    """
    Entraîne un classifieur EMG avec normalisation
    """

    # ---------- 1. Charger les données ----------
    # label_file = "train1_labels.csv" if "train" in file_name.lower() else "test1_labels.csv"

    # frame = inspect.currentframe().f_back
    # file_name = frame.f_globals.get(base64.b64decode("RU1HMl9maWxl").decode())
    # label_file = frame.f_globals.get(base64.b64decode("bGFiZWwyX2ZpbGU=").decode())

    global label_threshold
    print(f"file_name: {file_name}")
    label_file = file_name
    label_file = label_file.replace(".csv", "_labels.csv")
    print(f"file_name: {file_name}")
    print(f"label_file: {label_file}")

    visualize_sampling_filter(file_name, threshold=label_threshold, fs=fs)
    while ONLINE:
        # get threshold from user. pass yes if satisfied, else repeat
        user_input = input("Put threshold for label generation (or 'yes' to confirm): ")
        if user_input.lower() == 'yes':
            break
        else:
            try:
                label_threshold = float(user_input)
            except ValueError:
                print("Invalid input. Please enter a number or 'yes'.")
        visualize_sampling_filter(file_name, threshold=label_threshold, fs=fs)

    labels = generate_labels_from_data(
        file_name,
        window_size=window_size,
        output_label_file=label_file,
        threshold = label_threshold
    )

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


def test_classifier(classif, ch1, ch2, window_size, num_window, buzzer, file_name):
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

    buffer1 = []
    buffer2 = []
    cnt = 0
    labels = []

    # step_size = window_size // 2  # overlap 50%
    step_size = window_size

    print("Démarrage classification temps réel...")

    file = open(file_name, 'w', newline="")
    writer = csv.writer(file)
    feilds = ['voltage1 (V)', 'voltage2 (V)']
    writer.writerow(feilds)

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
    
    file.close()
    
    #save labels
    label_file = file_name.replace(".csv", "_predicted_labels.csv")
    with open(label_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'])
        for label in labels:
            writer.writerow([label])
    
    return labels