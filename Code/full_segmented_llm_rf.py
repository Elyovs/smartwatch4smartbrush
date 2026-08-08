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
test_file = "../data/3/new_train_set2.csv" 
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

def extract_features(window, df_len, current_idx):
    global_relative_position = (current_idx + window_size // 2) / df_len
    features = [global_relative_position]
    
    # 1. Absolute Variance & Range
    for col in sensor_cols:
        features.extend([window[col].mean(), window[col].std(), window[col].max() - window[col].min()])
        
    # 2. Normalized Variance (Movement Direction / Shape)
    sum_acc_std = window['ax'].std() + window['ay'].std() + window['az'].std() + 1e-6
    sum_gyr_std = window['gx'].std() + window['gy'].std() + window['gz'].std() + 1e-6
    features.extend([
        window['ax'].std() / sum_acc_std, window['ay'].std() / sum_acc_std, window['az'].std() / sum_acc_std,
        window['gx'].std() / sum_gyr_std, window['gy'].std() / sum_gyr_std, window['gz'].std() / sum_gyr_std
    ])
    
    # 3. Axis Cross-Correlations (Wrist Twist Mechanics)
    correlations = [
        window['ax'].corr(window['ay']), window['ax'].corr(window['az']), window['ay'].corr(window['az']),
        window['gx'].corr(window['gy']), window['gx'].corr(window['gz']), window['gy'].corr(window['gz'])
    ]
    features.extend(np.nan_to_num(correlations))
        
    # 4. Deltas (Rate of Change)
    for col in sensor_cols:
        delta_col = f'{col}_delta'
        features.extend([window[delta_col].mean(), window[delta_col].std(), window[delta_col].max() - window[delta_col].min()])
        
    # 5. Total Magnitudes
    mag_acc = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
    mag_gyro = np.sqrt(window['gx']**2 + window['gy']**2 + window['gz']**2)
    features.extend([mag_acc.std(), mag_acc.max() - mag_acc.min(), mag_gyro.std(), mag_gyro.max() - mag_gyro.min()])
    
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
            all_features.append(extract_features(df_copy.iloc[i : i + window_size], len(df_copy), i))
            all_labels.append(target_zone)
    return np.array(all_features), np.array(all_labels)

print("✂️ Extracting pure training features...")
X_train, y_train = load_pure_training_data(segmented_train_files)

print("🧠 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
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

# ==========================================
# 4. STAGE 2: LLM AGENT CLASSIFICATION
# ==========================================
print("\n🤖 STAGE 2: Activating LOCAL Ollama Domain Agent...")

df_copy = df_test[sensor_cols].copy()
for col in sensor_cols:
    df_copy[col] = df_copy[col].rolling(window=5, center=True, min_periods=1).mean()
    df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)

detected_sequence = []
classes = model.classes_
recent_history = []

all_window_true_labels = []
all_window_predictions = []

system_instruction = """
You are an expert AI spatial reasoning agent for a smart toothbrush. 
You are evaluating a sequence of brushing blocks that have been pre-segmented by physical transitions.

CRITICAL LAWS OF PHYSICS:
1. Because these blocks are separated by known physical transitions, each block represents a move to a NEW zone. 
2. Therefore, Block 2 CANNOT be the exact same zone as Block 1.
3. The user can move the brush anywhere during a transition. Do NOT force them to be adjacent.

Your job is to look at the Random Forest's statistical guesses and ensure no two adjacent blocks share the same zone.
Respond ONLY with a JSON object containing "predicted_zone" and "reasoning".
"""

for i in range(len(chunk_boundaries) - 1):
    start_idx = chunk_boundaries[i]
    end_idx = chunk_boundaries[i+1]
    full_chunk_len = end_idx - start_idx
    
    # Exclude edges to get the purest center data
    buffer = int(full_chunk_len * 0.1)
    clean_start = start_idx + buffer
    clean_end = end_idx - buffer
    
    chunk_df = df_copy.iloc[clean_start : clean_end]
    
    chunk_features = []
    for j in range(0, len(chunk_df) - window_size, step):
        chunk_features.append(extract_features(chunk_df.iloc[j : j + window_size], full_chunk_len, buffer + j))
    
    if not chunk_features:
        continue
        
    rf_probabilities = model.predict_proba(chunk_features)
    mean_probs = np.mean(rf_probabilities, axis=0)
    top_2_indices = np.argsort(mean_probs)[-2:][::-1]
    
    top_class = classes[top_2_indices[0]]
    top_prob = mean_probs[top_2_indices[0]] * 100
    runner_up = classes[top_2_indices[1]]
    runner_up_prob = mean_probs[top_2_indices[1]] * 100
    
    grav_x, grav_y, grav_z = round(np.mean([f[1] for f in chunk_features]), 2), round(np.mean([f[4] for f in chunk_features]), 2), round(np.mean([f[7] for f in chunk_features]), 2)
    
    domain_prompt = f"""
    Recent History (Last zone): {recent_history[-1] if recent_history else 'None (Start of session)'}
    Average Gravity Vector: X={grav_x}, Y={grav_y}, Z={grav_z}
    
    Random Forest Analysis for this block:
    1. {top_class} ({top_prob:.1f}%)
    2. {runner_up} ({runner_up_prob:.1f}%)
    
    Task: What is the true zone? If the RF's top choice is the EXACT same as the 'Recent History' zone, you MUST override it and pick the runner up. Otherwise, trust the top choice.
    """
    
    print(f"\n🤔 Analyzing Block {i+1}...")
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': domain_prompt}
            ],
            format='json'
        )
        agent_decision = json.loads(response['message']['content'])
        final_zone = agent_decision.get("predicted_zone", top_class)
        reason = agent_decision.get("reasoning", "No reasoning provided.")
    except Exception as e:
        final_zone = top_class
        reason = f"Ollama Error. Fallback to RF. ({str(e)})"

    # Failsafe in python just in case the local LLM hallucinates a non-existent zone
    if final_zone not in classes:
        final_zone = top_class
        reason = "LLM Hallucinated invalid zone. Fallback to RF."

    print(f"RF Suggested: {top_class} ({top_prob:.1f}%) | Runner up: {runner_up} ({runner_up_prob:.1f}%)")
    print(f"🤖 LLM Decided: {final_zone}")
    print(f"   ↳ Reason: {reason}")
    
    detected_sequence.append(final_zone)
    recent_history.append(final_zone)

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