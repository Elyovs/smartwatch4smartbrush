import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import re
from collections import Counter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import json
import ollama
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 1. FILE CONFIGURATION (Dynamic LOSO Split)
# ==========================================
all_segmented_files = [
    # Session 1
    "../data/3/upper_front_buccal_s1.csv", "../data/3/upper_left_buccal_s1.csv", "../data/3/upper_right_buccal_s1.csv", "../data/3/upper_left_occlusal_s1.csv", "../data/3/upper_right_occlusal_s1.csv", "../data/3/upper_left_lingual_s1.csv", "../data/3/upper_front_lingual_s1.csv", "../data/3/upper_right_lingual_s1.csv",
    # Session 2
    "../data/3/upper_front_buccal_s2.csv", "../data/3/upper_left_buccal_s2.csv", "../data/3/upper_right_buccal_s2.csv", "../data/3/upper_left_occlusal_s2.csv", "../data/3/upper_right_occlusal_s2.csv", "../data/3/upper_left_lingual_s2.csv", "../data/3/upper_front_lingual_s2.csv", "../data/3/upper_right_lingual_s2.csv",
    # Session 3
    "../data/3/upper_front_buccal_s3.csv", "../data/3/upper_left_buccal_s3.csv", "../data/3/upper_right_buccal_s3.csv", "../data/3/upper_left_occlusal_s3.csv", "../data/3/upper_right_occlusal_s3.csv", "../data/3/upper_left_lingual_s3.csv", "../data/3/upper_front_lingual_s3.csv", "../data/3/upper_right_lingual_s3.csv",
    # Session 4
    "../data/3/upper_front_buccal_s4.csv", "../data/3/upper_left_buccal_s4.csv", "../data/3/upper_right_buccal_s4.csv", "../data/3/upper_left_occlusal_s4.csv", "../data/3/upper_right_occlusal_s4.csv", "../data/3/upper_left_lingual_s4.csv", "../data/3/upper_front_lingual_s4.csv", "../data/3/upper_right_lingual_s4.csv",
    # Session 5
    "../data/3/upper_front_buccal_s5.csv", "../data/3/upper_left_buccal_s5.csv", "../data/3/upper_right_buccal_s5.csv", "../data/3/upper_left_occlusal_s5.csv", "../data/3/upper_right_occlusal_s5.csv", "../data/3/upper_left_lingual_s5.csv", "../data/3/upper_front_lingual_s5.csv", "../data/3/upper_right_lingual_s2.csv",
    # Session 6
    "../data/3/upper_front_buccal_s6.csv", "../data/3/upper_left_buccal_s6.csv", "../data/3/upper_right_buccal_s6.csv", "../data/3/upper_left_occlusal_s6.csv", "../data/3/upper_right_occlusal_s6.csv", "../data/3/upper_left_lingual_s6.csv", "../data/3/upper_front_lingual_s6.csv", "../data/3/upper_right_lingual_s6.csv",
    # Session 7
    "../data/3/upper_front_buccal_s7.csv", "../data/3/upper_left_buccal_s7.csv", "../data/3/upper_right_buccal_s7.csv", "../data/3/upper_left_occlusal_s7.csv", "../data/3/upper_right_occlusal_s7.csv", "../data/3/upper_left_lingual_s7.csv", "../data/3/upper_front_lingual_s7.csv", "../data/3/upper_right_lingual_s7.csv"
]

# TARGETING THE SPECIFIC FILE
test_file = "../data/3/new_train_set8.csv"
sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
expected_sequence = ["front_buccal", "left_buccal", "right_buccal", "left_occlusal", "right_occlusal", "left_lingual", "front_lingual", "right_lingual"]

# Prevent Data Leakage (LOSO)
session_match = re.search(r'set(\d+)\.csv', test_file)
test_session_id = session_match.group(1) if session_match else "unknown"
segmented_train_files = [f for f in all_segmented_files if not f.endswith(f"_s{test_session_id}.csv")]

print("Configuration:")
print(f"   Test File: {test_file} (Session {test_session_id})")
print(f"   Training Files: {len(segmented_train_files)} (Session {test_session_id} excluded to prevent data leakage)\n")

# ==========================================
# 2. FEATURE EXTRACTION (Advanced Math)
# ==========================================
window_size = 40
step = 20

def extract_features(window):
    window = window.copy()

    for col in sensor_cols:
        window[col] = window[col] - window[col].mean()

    features = []

    # mean, std, range, median, RMS, Q25, Q75
    for col in sensor_cols:
        x = window[col]
        features.extend([
            x.mean(),
            x.std(),
            x.max() - x.min(),
            x.median(),
            np.sqrt(np.mean(x**2)),
            np.percentile(x, 25),
            np.percentile(x, 75)
        ])

    # Relative variance
    sum_acc = (
        window['ax'].std() +
        window['ay'].std() +
        window['az'].std() + 1e-6
    )

    sum_gyr = (
        window['gx'].std() +
        window['gy'].std() +
        window['gz'].std() + 1e-6
    )

    features.extend([
        window['ax'].std() / sum_acc,
        window['ay'].std() / sum_acc,
        window['az'].std() / sum_acc,

        window['gx'].std() / sum_gyr,
        window['gy'].std() / sum_gyr,
        window['gz'].std() / sum_gyr,
    ])

    # Correlations
    corr = [
        window['ax'].corr(window['ay']),
        window['ax'].corr(window['az']),
        window['ay'].corr(window['az']),
        window['gx'].corr(window['gy']),
        window['gx'].corr(window['gz']),
        window['gy'].corr(window['gz'])
    ]

    features.extend(np.nan_to_num(corr))

    # Delta features
    for col in sensor_cols:
        delta = f"{col}_delta"
        features.extend([
            window[delta].mean(),
            window[delta].std(),
            window[delta].max() - window[delta].min()
        ])

    # Magnitude
    raw = window.copy()

    mag_acc = np.sqrt(raw.ax**2 + raw.ay**2 + raw.az**2)
    mag_gyro = np.sqrt(raw.gx**2 + raw.gy**2 + raw.gz**2)

    features.extend([
        mag_acc.mean(),
        mag_acc.std(),
        mag_acc.max() - mag_acc.min(),

        mag_gyro.mean(),
        mag_gyro.std(),
        mag_gyro.max() - mag_gyro.min()
    ])

    # Orientation feature (gravity direction)
    # BUG FIX: original code computed this normalized vector twice, discarding
    # the first (harmless but wasteful) computation. Cleaned up to a single pass.
    raw_acc = window[['ax', 'ay', 'az']].copy()
    gravity = raw_acc.mean()
    g = np.array([gravity['ax'], gravity['ay'], gravity['az']])
    norm = np.linalg.norm(g)
    if norm > 1e-6:
        g = g / norm
    else:
        g = np.zeros(3)

    features.extend(g.tolist())

    return features

def load_pure_training_data(file_list):
    all_features, all_labels = [], []
    for p in file_list:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        target_zone = "_".join(os.path.basename(p).replace(".csv", "").split("_")[1:-1])

        df_copy = df[sensor_cols].copy()
        for col in sensor_cols:
            df_copy[col] = df_copy[col].rolling(window=5, center=True, min_periods=1).mean()
            df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)

        for i in range(0, len(df_copy) - window_size, step):
            all_features.append(extract_features(df_copy.iloc[i:i + window_size]))
            all_labels.append(target_zone)
    return np.array(all_features), np.array(all_labels)

print("Extracting pure training features...")
X_train, y_train = load_pure_training_data(segmented_train_files)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
pca = PCA(
    n_components=0.98,
    random_state=42
)
X_train = pca.fit_transform(X_train)

print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": np.arange(len(model.feature_importances_)),
    "Importance": model.feature_importances_
})
importance = importance.sort_values("Importance", ascending=False)
print(importance.head(20))

# ==========================================
# 3. STAGE 1: AUTO-SEGMENTATION (STEP-EDGE / PLATEAU DETECTION)
# ==========================================
print("\nSTAGE 1: Slicing via Step-Edge Detection (Mimicking the Human Eye)...")

df_test = pd.read_csv(test_file)
total_rows = len(df_test)

n_zones = len(expected_sequence)
n_expected_transitions = n_zones - 1
avg_zone_length = total_rows / n_zones

# 1. Light low-pass filter to remove vibration
smooth_window = 50 
g_x = df_test['ax'].rolling(window=smooth_window, center=True, min_periods=1).mean()
g_y = df_test['ay'].rolling(window=smooth_window, center=True, min_periods=1).mean()
g_z = df_test['az'].rolling(window=smooth_window, center=True, min_periods=1).mean()

# 2. The "Human Eye" Filter (Difference of adjacent blocks)
W = int(avg_zone_length * 0.4) 

left_mean_x = g_x.rolling(window=W, min_periods=1).mean()
left_mean_y = g_y.rolling(window=W, min_periods=1).mean()
left_mean_z = g_z.rolling(window=W, min_periods=1).mean()

right_mean_x = g_x.shift(-W).rolling(window=W, min_periods=1).mean()
right_mean_y = g_y.shift(-W).rolling(window=W, min_periods=1).mean()
right_mean_z = g_z.shift(-W).rolling(window=W, min_periods=1).mean()

# 3. Measure the distance between the two plateaus
step_diff_x = right_mean_x - left_mean_x
step_diff_y = right_mean_y - left_mean_y
step_diff_z = right_mean_z - left_mean_z

transition_metric = np.sqrt(step_diff_x**2 + step_diff_y**2 + step_diff_z**2).fillna(0)
transition_metric = transition_metric.rolling(window=50, center=True, min_periods=1).mean().fillna(0)

# 4. Candidate Peak Finding
max_energy = np.max(transition_metric)
dynamic_prominence = max_energy * 0.01 
min_distance = int(avg_zone_length * 0.25) 

raw_peaks, properties = find_peaks(
    transition_metric,
    distance=min_distance,
    prominence=dynamic_prominence
)

# 5. REMOVE MATHEMATICAL EDGE ARTIFACTS
# The rolling windows run out of data at the edges, creating massive fake peaks at exactly W.
# We filter out any peaks that fall inside this blind spot.
edge_buffer = int(W * 1.2)
valid_mask = (raw_peaks > edge_buffer) & (raw_peaks < (total_rows - edge_buffer))

filtered_peaks = raw_peaks[valid_mask]
filtered_prominences = properties['prominences'][valid_mask]
candidate_peaks = filtered_peaks.copy()

# 6. Enforce EXACTLY N-1 Transitions (Keep the strongest valid peaks)
if len(filtered_peaks) > n_expected_transitions:
    top_order = np.argsort(filtered_prominences)[::-1][:n_expected_transitions]
    valid_transitions = sorted(filtered_peaks[top_order])
elif len(filtered_peaks) < n_expected_transitions:
    print(f"WARNING: only found {len(filtered_peaks)} candidate transitions, expected {n_expected_transitions}.")
    valid_transitions = sorted(filtered_peaks)
else:
    valid_transitions = sorted(filtered_peaks)

chunk_boundaries = [0] + valid_transitions + [total_rows]

print(f"Found {len(candidate_peaks)} valid candidate transitions -> kept the "
      f"{len(valid_transitions)} strongest, matching {n_zones} expected zones.")

# 7. Visualization
plt.figure(figsize=(12, 4))
plt.plot(transition_metric, label="Plateau Shift Distance (Human Eye Filter)", color="purple")
plt.axhline(y=dynamic_prominence, color='green', linestyle=':', label="Prominence Threshold")

for cp in candidate_peaks:
    if cp not in valid_transitions:
        plt.axvline(x=cp, color='gray', linestyle=':', linewidth=1, alpha=0.6)
for cp in valid_transitions:
    plt.axvline(x=cp, color='red', linestyle='--', linewidth=2)

# Highlight the ignored edge zones
plt.axvspan(0, edge_buffer, color='gray', alpha=0.2, label="Ignored Edge Artifacts")
plt.axvspan(total_rows - edge_buffer, total_rows, color='gray', alpha=0.2)

plt.title(f"Auto-Slicer Cuts for {test_file}\n"
          f"({len(valid_transitions)}/{n_expected_transitions} transitions kept)")
plt.xlabel("Rows (Time)")
plt.ylabel("Difference Between Left & Right Blocks")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# STAGE 2: CALIBRATED RANDOM FOREST
# ==========================================
print("\nSTAGE 2: Calibrated Random Forest Classification...")

df_copy = df_test[sensor_cols].copy()

for col in sensor_cols:
    df_copy[col] = df_copy[col].rolling(window=5, center=True, min_periods=1).mean()
    df_copy[f"{col}_delta"] = df_copy[col].diff().fillna(0)

# --------------------------------------------------
# STEP 1: EXTRACT FEATURES FOR EACH BLOCK
# --------------------------------------------------

block_features = []
n_features_expected = None

for block_idx in range(len(chunk_boundaries) - 1):
    start_idx = chunk_boundaries[block_idx]
    end_idx = chunk_boundaries[block_idx + 1]
    block_len = end_idx - start_idx

    buffer = max(80, int(block_len * 0.20))
    clean_start = start_idx + buffer
    clean_end = end_idx - buffer

    if clean_end <= clean_start:
        clean_start = start_idx
        clean_end = end_idx

    chunk_df = df_copy.iloc[clean_start:clean_end]

    features = []
    for j in range(0, len(chunk_df) - window_size, step):
        feat = extract_features(chunk_df.iloc[j:j + window_size])

        # FIX: the "81" was a hardcoded magic number that silently goes stale
        # if extract_features() is ever edited. Derive it once instead.
        if n_features_expected is None:
            n_features_expected = len(feat)
        elif len(feat) != n_features_expected:
            raise ValueError(
                f"Feature mismatch in Block {block_idx + 1}: "
                f"expected {n_features_expected}, got {len(feat)}"
            )

        features.append(feat)

    if len(features) == 0:
        block_features.append(None)
    else:
        block_features.append(np.asarray(features, dtype=float))
        print(f"Block {block_idx + 1}: {len(features)} windows x {len(features[0])} features")

# --------------------------------------------------
# STEP 2: BLOCK 1 = CALIBRATION REFERENCE
# --------------------------------------------------

print("\nCALIBRATION")
print("Block 1 is assumed to be FRONT_BUCCAL.")

calibration_features = block_features[0]

if calibration_features is None:
    raise ValueError("Could not extract enough features from Block 1 for calibration.")

print(f"Calibration windows: {len(calibration_features)}")
print(f"Calibration feature shape: {calibration_features.shape}")

calibration_profile = np.mean(calibration_features, axis=0)
calibration_std = np.std(calibration_features, axis=0) + 1e-6

print("Front-buccal calibration established.")

# --------------------------------------------------
# STEP 3: CLASSIFY EACH BLOCK
# --------------------------------------------------

detected_sequence = ["front_buccal"]
classes = model.classes_

for block_idx in range(1, len(block_features)):
    features = block_features[block_idx]

    if features is None:
        detected_sequence.append("unknown")
        continue

    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    rf_probabilities = model.predict_proba(features_pca)
    mean_probs = np.mean(rf_probabilities, axis=0)
    top_indices = np.argsort(mean_probs)[::-1]

    print(f"\nBlock {block_idx + 1}")
    print("RF probabilities:")
    for idx in top_indices[:3]:
        print(f"   {classes[idx]:20s} {mean_probs[idx]:.2%}")

    rf_prediction = classes[np.argmax(mean_probs)]

    block_profile = np.mean(features, axis=0)
    normalized_difference = np.abs(block_profile - calibration_profile) / calibration_std
    calibration_distance = np.mean(normalized_difference)

    print(f"Calibration distance: {calibration_distance:.3f}")

    final_zone = rf_prediction
    confidence = np.max(mean_probs)

    print(f"Prediction : {final_zone}")
    print(f"Confidence : {confidence:.2%}")

    detected_sequence.append(final_zone)

# --------------------------------------------------
# STEP 4: FINAL SEQUENCE
# --------------------------------------------------

print("\n==========================================")
print("        CALIBRATED DETECTED SEQUENCE")
print("==========================================")
print(" -> ".join(detected_sequence))

# ==========================================
# 5. FULL CONTINUOUS RECONSTRUCTION
# ==========================================
print("\n==========================================")
print("     CONTINUOUS FILE RECONSTRUCTION")
print("==========================================")

all_window_true_labels = []
all_window_predictions = []

transition_radius = window_size * 2

for i in range(0, len(df_test) - window_size, step):
    midpoint = i + window_size // 2

    is_transition = False
    # BUG FIX: `top_7_transitions` was never defined anywhere in the original
    # script -> guaranteed NameError as soon as this loop ran. It should be
    # the final selected transitions from Stage 1.
    for valley in valid_transitions:
        if abs(midpoint - valley) <= transition_radius:
            is_transition = True
            break

    if is_transition:
        all_window_true_labels.append("transition")
        all_window_predictions.append("transition")
    else:
        block_idx = 0
        for b_idx in range(len(chunk_boundaries) - 1):
            if chunk_boundaries[b_idx] <= midpoint <= chunk_boundaries[b_idx + 1]:
                block_idx = b_idx
                break

        expected_zone = expected_sequence[block_idx] if block_idx < len(expected_sequence) else "unknown"
        predicted_zone = detected_sequence[block_idx] if block_idx < len(detected_sequence) else "unknown"

        all_window_true_labels.append(expected_zone)
        all_window_predictions.append(predicted_zone)

# ==========================================
# 6. FINAL METRICS & CONFUSION MATRIX
# ==========================================
print("INSTRUCTED ORDER:\n -> ".join(expected_sequence))
print("\nDETECTED ORDER:\n -> ".join(detected_sequence))

window_accuracy = accuracy_score(all_window_true_labels, all_window_predictions) * 100
print(f"\nFULL CONTINUOUS FILE ACCURACY: {window_accuracy:.1f}%\n")

matrix_labels = expected_sequence + ["transition"]

print("\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(all_window_true_labels, all_window_predictions, labels=matrix_labels, zero_division=0))

print("\nGenerating confusion matrix...")
cm = confusion_matrix(all_window_true_labels, all_window_predictions, labels=matrix_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=matrix_labels)

fig, ax = plt.subplots(figsize=(12, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title(f"Continuous Hybrid Pipeline Confusion Matrix\nTest File: {test_file}", fontsize=14, pad=20)
plt.tight_layout()
plt.show()