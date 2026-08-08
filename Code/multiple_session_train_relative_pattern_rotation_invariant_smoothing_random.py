import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import random

all_files = [
    "../data/2/upper_front_buccal_s1.csv", "../data/2/upper_front_lingual_s1.csv",
    "../data/2/upper_left_buccal_s1.csv", "../data/2/upper_left_lingual_s1.csv",
    "../data/2/upper_left_occlusal_s1.csv", "../data/2/upper_right_buccal_s1.csv",
    "../data/2/upper_right_lingual_s1.csv", "../data/2/upper_right_occlusal_s1.csv",
    
    "../data/2/upper_front_buccal_s2.csv", "../data/2/upper_front_lingual_s2.csv",
    "../data/2/upper_left_buccal_s2.csv", "../data/2/upper_left_lingual_s2.csv",
    "../data/2/upper_left_occlusal_s2.csv", "../data/2/upper_right_buccal_s2.csv",
    "../data/2/upper_right_lingual_s2.csv", "../data/2/upper_right_occlusal_s2.csv",
    
    "../data/2/upper_front_buccal_s4.csv", "../data/2/upper_front_lingual_s4.csv",
    "../data/2/upper_left_buccal_s4.csv", "../data/2/upper_left_lingual_s4.csv",
    "../data/2/upper_left_occlusal_s4.csv", "../data/2/upper_right_buccal_s4.csv",
    "../data/2/upper_right_lingual_s4.csv", "../data/2/upper_right_occlusal_s4.csv"
]

# --- SHUFFLE LOGIC: ENFORCE EXACT 2-TRAIN / 1-TEST SESSION SEPARATION ---
def split_files_by_random_sessions(file_list):
    """
    Groups files by their natural session tags, shuffles them randomly, 
    and returns exactly 2 sessions for training and 1 session for testing.
    """
    session_groups = {}
    for f in file_list:
        base = os.path.basename(f).replace(".csv", "")
        session_id = base.split("_")[-1]  # Extracts 's1', 's2', or 's4'
        
        if session_id not in session_groups:
            session_groups[session_id] = []
        session_groups[session_id].append(f)
    
    # Get unique session strings and shuffle randomly
    unique_sessions = list(session_groups.keys())
    random.shuffle(unique_sessions)
    
    # Enforce clear 2-and-1 split indices
    train_sessions = unique_sessions[:2]
    test_sessions = unique_sessions[2:]
    
    train_files = []
    for sess in train_sessions:
        train_files.extend(session_groups[sess])
        
    test_files = []
    for sess in test_sessions:
        test_files.extend(session_groups[sess])
        
    print(f"🎲 Randomized Session Configuration:")
    print(f"   Selected Training Sessions (2): {train_sessions} ({len(train_files)} files)")
    print(f"   Selected Testing Sessions  (1): {test_sessions} ({len(test_files)} files)\n")
    
    return train_files, test_files

# Run the fixed session randomizer
train_files, test_files = split_files_by_random_sessions(all_files)

# 1. Load and Label Data
# train_files = [
#     "../data/2/upper_front_buccal_s1.csv",
#     "../data/2/upper_front_lingual_s1.csv",
#     "../data/2/upper_left_buccal_s1.csv",
#     "../data/2/upper_left_lingual_s1.csv",
#     "../data/2/upper_left_occlusal_s1.csv",
#     "../data/2/upper_right_buccal_s1.csv",
#     "../data/2/upper_right_lingual_s1.csv",
#     "../data/2/upper_right_occlusal_s1.csv",
#     "../data/2/upper_front_buccal_s2.csv",
#     "../data/2/upper_front_lingual_s2.csv",
#     "../data/2/upper_left_buccal_s2.csv",
#     "../data/2/upper_left_lingual_s2.csv",
#     "../data/2/upper_left_occlusal_s2.csv",
#     "../data/2/upper_right_buccal_s2.csv",
#     "../data/2/upper_right_lingual_s2.csv",
#     "../data/2/upper_right_occlusal_s2.csv"
# ]

# test_files = [
#     "../data/2/upper_front_buccal_s4.csv",
#     "../data/2/upper_front_lingual_s4.csv",
#     "../data/2/upper_left_buccal_s4.csv",
#     "../data/2/upper_left_lingual_s4.csv",
#     "../data/2/upper_left_occlusal_s4.csv",
#     "../data/2/upper_right_buccal_s4.csv",
#     "../data/2/upper_right_lingual_s4.csv",
#     "../data/2/upper_right_occlusal_s4.csv"
# ]

def load_and_label_by_file(file_list):
    all_features = []
    all_labels = []
    
    window_size = 40
    step = 20
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for f in file_list:
        df = pd.read_csv(f)
        total_rows = len(df) # Get total length of this session
        
        # Calculate row-to-row deltas
        df_copy = df[sensor_cols].copy()
        for col in sensor_cols:
            df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)
            
        base_name = os.path.basename(f).replace(".csv", "")
        clean_label = "_".join(base_name.split("_")[:-1]) 
        
        for i in range(0, len(df_copy) - window_size, step):
            window = df_copy.iloc[i : i + window_size]
            
            # --- THE BREAKTHROUGH FEATURE: RELATIVE TIMELINE POSITION ---
            # Finds exactly where this window sits in the timeline (0.0 to 1.0)
            relative_position = (i + window_size // 2) / total_rows
            
            # Start our feature list with the timeline context
            features = [relative_position]
            
            # Extract standard pattern features (Vibrations and Stroke Sizes)
            for col in sensor_cols:
                features.extend([
                    window[col].std(),                     
                    window[col].max() - window[col].min()  
                ])
                
            # Extract delta wave patterns (Velocity shifts)
            for col in sensor_cols:
                delta_col = f'{col}_delta'
                features.extend([
                    window[delta_col].mean(),              
                    window[delta_col].std(),               
                    window[delta_col].max() - window[delta_col].min() 
                ])
            
            # Scale-invariant total magnitude energies
            mag_acc = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
            mag_gyro = np.sqrt(window['gx']**2 + window['gy']**2 + window['gz']**2)
            features.extend([
                mag_acc.std(), mag_acc.max() - mag_acc.min(),
                mag_gyro.std(), mag_gyro.max() - mag_gyro.min()
            ])
            
            all_features.append(features)
            all_labels.append(clean_label)
            
    return np.array(all_features), np.array(all_labels)

# Extract relative features
X_train, y_train = load_and_label_by_file(train_files)
X_test, y_test = load_and_label_by_file(test_files)

# Train using class_weight='balanced' to handle unequal segment lengths gracefully
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("--- Evaluation on Unseen Test Session ---")
print(classification_report(y_test, model.predict(X_test)))

# 1. Get the raw window-by-window string predictions from your model
raw_predictions = model.predict(X_test)

# 2. Apply a rolling majority vote over a window of 7 consecutive predictions
smoothed_predictions = []
filter_size = 7

for i in range(len(raw_predictions)):
    # Create a neighborhood around the current prediction
    start_idx = max(0, i - filter_size // 2)
    end_idx = min(len(raw_predictions), i + filter_size // 2 + 1)
    neighborhood = raw_predictions[start_idx:end_idx]
    
    # --- FIXED LOGIC USING COUNTER ---
    # Counter converts neighborhood to: {'upper_left_buccal': 5, 'upper_left_occlusal': 2}
    # .most_common(1)[0][0] instantly extracts the string with the highest count
    majority_vote = Counter(neighborhood).most_common(1)[0][0]
    smoothed_predictions.append(majority_vote)

# 3. Evaluate the smoothed results instead of the raw ones!
print("--- Evaluation AFTER Prediction Smoothing ---")
print(classification_report(y_test, smoothed_predictions))

# Confusion Matrix Visualizing
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix: Pattern-Based Cross-Session Generalization")
plt.tight_layout()
plt.show()