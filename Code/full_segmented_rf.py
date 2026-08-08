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
    "../data/3/upper_front_buccal_s4.csv", "../data/3/upper_left_buccal_s4.csv", "../data/3/upper_right_buccal_s4.csv", "../data/3/upper_left_occlusal_s4.csv", "../data/3/upper_right_occlusal_s4.csv", "../data/3/upper_left_lingual_s4.csv", "../data/3/upper_front_lingual_s4.csv", "../data/3/upper_right_lingual_s4.csv"
]

# TARGETING THE SPECIFIC FILE
test_file = "../data/3/new_train_set4.csv" 
sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
expected_sequence = ["front_buccal", "left_buccal", "right_buccal", "left_occlusal", "right_occlusal", "left_lingual", "front_lingual", "right_lingual"]

# Prevent Data Leakage (LOSO)
session_match = re.search(r'set(\d+)\.csv', test_file)
test_session_id = session_match.group(1) if session_match else "unknown"
segmented_train_files = [f for f in all_segmented_files if not f.endswith(f"_s{test_session_id}.csv")]

print("🎲 Configuration:")
print(f"   Test File: {test_file} (Session {test_session_id})")
print(f"   Training Files: {len(segmented_train_files)} (Session {test_session_id} excluded to prevent data leakage)\n")

# ==========================================
# 2. FEATURE EXTRACTION (Advanced Math)
# ==========================================
window_size = 40
step = 20

def extract_features(window):
    features = []

    # Mean / Std / Range
    for col in sensor_cols:
        features.extend([
            window[col].mean(),
            window[col].std(),
            window[col].max() - window[col].min()
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
        window['ax'].std()/sum_acc,
        window['ay'].std()/sum_acc,
        window['az'].std()/sum_acc,

        window['gx'].std()/sum_gyr,
        window['gy'].std()/sum_gyr,
        window['gz'].std()/sum_gyr,
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
            window[delta].max()-window[delta].min()
        ])

    # Magnitude
    mag_acc = np.sqrt(window.ax**2 + window.ay**2 + window.az**2)
    mag_gyro = np.sqrt(window.gx**2 + window.gy**2 + window.gz**2)

    features.extend([
        mag_acc.mean(),
        mag_acc.std(),
        mag_acc.max()-mag_acc.min(),

        mag_gyro.mean(),
        mag_gyro.std(),
        mag_gyro.max()-mag_gyro.min()
    ])

    return features

def load_pure_training_data(file_list):
    all_features, all_labels = [], []
    for p in file_list:
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        target_zone = "_".join(os.path.basename(p).replace(".csv", "").split("_")[1:-1])
        
        df_copy = df[sensor_cols].copy()
        for col in sensor_cols:
            df_copy[col] = df_copy[col].rolling(window=5, center=True, min_periods=1).mean()
            df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)
            
        for i in range(0, len(df_copy) - window_size, step):
            all_features.append(extract_features(df_copy.iloc[i:i+window_size]))
            all_labels.append(target_zone)
    return np.array(all_features), np.array(all_labels)

print("✂️ Extracting pure training features...")
X_train, y_train = load_pure_training_data(segmented_train_files)

print("🧠 Training Random Forest...")
# model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ==========================================
# 3. STAGE 1: AUTO-SEGMENTATION (THE SLICER)
# ==========================================
print("\n🔪 STAGE 1: Slicing the continuous file at transition valleys...")

df_test = pd.read_csv(test_file)

accel_mag = np.sqrt(df_test['ax']**2 + df_test['ay']**2 + df_test['az']**2)
movement_intensity = accel_mag.rolling(window=40, center=True, min_periods=1).std()
smoothed_envelope = movement_intensity.rolling(window=400, center=True, min_periods=1).mean().fillna(0).values

# Sort by PROMINENCE and use a smaller distance to perfectly catch quick transitions
valleys, properties = find_peaks(-smoothed_envelope, distance=800, prominence=0.01)

if len(valleys) >= 7:
    top_7_idx = np.argsort(properties["prominences"])[-7:]
    top_7_transitions = sorted(valleys[top_7_idx])
else:
    top_7_transitions = sorted(valleys)

chunk_boundaries = [0] + top_7_transitions + [len(df_test)]
print(f"✅ Found {len(top_7_transitions)} transitions. Sliced file into {len(chunk_boundaries)-1} blocks.")

plt.figure(figsize=(12, 4))
plt.plot(smoothed_envelope, label="Smoothed Brushing Envelope", color="blue")
for cp in top_7_transitions:
    plt.axvline(x=cp, color='red', linestyle='--', linewidth=2)
plt.title(f"Auto-Slicer Cuts for {test_file}")
plt.xlabel("Rows (Time)")
plt.ylabel("Movement Intensity")
plt.legend()
plt.tight_layout()
print("\n>>> CLOSE THE MATPLOTLIB GRAPH TO CONTINUE SCRIPT <<<")
plt.show()

print("\n🤖 STAGE 2: Random Forest Chunk Classification...")

df_copy = df_test[sensor_cols].copy()

for col in sensor_cols:
    df_copy[col] = df_copy[col].rolling(
        window=5,
        center=True,
        min_periods=1
    ).mean()

    df_copy[f"{col}_delta"] = (
        df_copy[col].diff().fillna(0)
    )

detected_sequence = []

for block_idx in range(len(chunk_boundaries)-1):
    start_idx = chunk_boundaries[block_idx]
    end_idx = chunk_boundaries[block_idx+1]

    block_len = end_idx - start_idx

    # Trim transitions
    buffer = max(80, int(block_len * 0.20))

    clean_start = start_idx + buffer
    clean_end = end_idx - buffer

    if clean_end <= clean_start:
        clean_start = start_idx
        clean_end = end_idx

    chunk_df = df_copy.iloc[clean_start:clean_end]

    chunk_features = []

    for j in range(0, len(chunk_df)-window_size,step):
        feat = extract_features(chunk_df.iloc[j:j+window_size])
        chunk_features.append(feat)

    if len(chunk_features) == 0:
        detected_sequence.append("unknown")
        continue

    chunk_features = np.array(chunk_features)

    # Predict every window
    preds = model.predict(chunk_features)

    # Majority voting
    vote_counter = Counter(preds)

    final_zone = vote_counter.most_common(1)[0][0]

    confidence = (vote_counter[final_zone] / len(preds))

    print(f"\nBlock {block_idx+1}")
    print(vote_counter)
    print(f"Prediction : {final_zone}")
    print(f"Confidence: {confidence:.2%}")

    detected_sequence.append(final_zone)

# ==========================================
# 5. FULL CONTINUOUS RECONSTRUCTION 
# ==========================================
print("\n==========================================")
print("     CONTINUOUS FILE RECONSTRUCTION")
print("==========================================")
# Now we map the entire file (including transitions) using the Slicer's exact coordinates!

all_window_true_labels = []
all_window_predictions = []

# A transition usually lasts ~2 to 3 seconds. 
# 100 rows at 50Hz = 2 seconds in each direction (4 seconds total gap)
transition_radius = 100 

for i in range(0, len(df_test) - window_size, step):
    midpoint = i + window_size // 2
    
    # 1. Is this window near a Slicer Valley?
    is_transition = False
    for valley in top_7_transitions:
        if abs(midpoint - valley) <= transition_radius:
            is_transition = True
            break
            
    if is_transition:
        # If it's near a valley, it is strictly a transition
        all_window_true_labels.append("transition")
        all_window_predictions.append("transition")
    else:
        # 2. If it's not a transition, which Slicer Block is it inside?
        block_idx = 0
        for b_idx in range(len(chunk_boundaries) - 1):
            if chunk_boundaries[b_idx] <= midpoint <= chunk_boundaries[b_idx+1]:
                block_idx = b_idx
                break
                
        expected_zone = expected_sequence[block_idx] if block_idx < len(expected_sequence) else "unknown"
        predicted_zone = detected_sequence[block_idx] if block_idx < len(detected_sequence) else "unknown"
        
        all_window_true_labels.append(expected_zone)
        all_window_predictions.append(predicted_zone)


# ==========================================
# 6. FINAL METRICS & 9x9 CONFUSION MATRIX
# ==========================================
print("📋 INSTRUCTED ORDER:\n -> ".join(expected_sequence))
print("\n🤖 DETECTED ORDER:\n -> ".join(detected_sequence))

window_accuracy = accuracy_score(all_window_true_labels, all_window_predictions) * 100
print(f"\n🎯 FULL CONTINUOUS FILE ACCURACY: {window_accuracy:.1f}%\n")

# We add 'transition' to the expected sequence so it shows up in the matrix!
matrix_labels = expected_sequence + ["transition"]

print("\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(all_window_true_labels, all_window_predictions, labels=matrix_labels, zero_division=0))

print("\n📊 Generating 9x9 Confusion Matrix...")
cm = confusion_matrix(all_window_true_labels, all_window_predictions, labels=matrix_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=matrix_labels)

fig, ax = plt.subplots(figsize=(12, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title(f"Continuous Hybrid Pipeline (9x9) Confusion Matrix\nTest File: {test_file}", fontsize=14, pad=20)
plt.tight_layout()
plt.show()