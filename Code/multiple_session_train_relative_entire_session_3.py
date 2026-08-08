import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import random
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

# Master source list grouped cleanly by natural session strings
all_files = [
    "../data/3/upper_front_buccal_s1.csv", "../data/3/upper_front_lingual_s1.csv",
    "../data/3/upper_left_buccal_s1.csv", "../data/3/upper_left_lingual_s1.csv",
    "../data/3/upper_left_occlusal_s1.csv", "../data/3/upper_right_buccal_s1.csv",
    "../data/3/upper_right_lingual_s1.csv", "../data/3/upper_right_occlusal_s1.csv",
    
    "../data/3/upper_front_buccal_s2.csv", "../data/3/upper_front_lingual_s2.csv",
    "../data/3/upper_left_buccal_s2.csv", "../data/3/upper_left_lingual_s2.csv",
    "../data/3/upper_left_occlusal_s2.csv", "../data/3/upper_right_buccal_s2.csv",
    "../data/3/upper_right_lingual_s2.csv", "../data/3/upper_right_occlusal_s2.csv",

    "../data/3/upper_front_buccal_s3.csv", "../data/3/upper_front_lingual_s3.csv",
    "../data/3/upper_left_buccal_s3.csv", "../data/3/upper_left_lingual_s3.csv",
    "../data/3/upper_left_occlusal_s3.csv", "../data/3/upper_right_buccal_s3.csv",
    "../data/3/upper_right_lingual_s3.csv", "../data/3/upper_right_occlusal_s3.csv",
    
    "../data/3/upper_front_buccal_s4.csv", "../data/3/upper_front_lingual_s4.csv",
    "../data/3/upper_left_buccal_s4.csv", "../data/3/upper_left_lingual_s4.csv",
    "../data/3/upper_left_occlusal_s4.csv", "../data/3/upper_right_buccal_s4.csv",
    "../data/3/upper_right_lingual_s4.csv", "../data/3/upper_right_occlusal_s4.csv"
]

def split_files_by_random_sessions(file_list):
    session_groups = {}
    for f in file_list:
        base = os.path.basename(f).replace(".csv", "")
        session_id = base.split("_")[-1]
        if session_id not in session_groups:
            session_groups[session_id] = []
        session_groups[session_id].append(f)
        
    unique_sessions = list(session_groups.keys())
    random.shuffle(unique_sessions)
    
    # train_sessions = unique_sessions[:2]
    # test_sessions = unique_sessions[2:]
    train_sessions = unique_sessions[:3]
    test_sessions = unique_sessions[3:]
    
    train_files = []
    for sess in train_sessions:
        train_files.extend(session_groups[sess])
    test_files = []
    for sess in test_sessions:
        test_files.extend(session_groups[sess])
        
    print(f"🎲 Randomized Session Configuration:")
    print(f"   Training Sessions (2): {train_sessions}")
    print(f"   Testing Sessions  (1): {test_sessions}\n")
    return train_files, test_files

train_files, test_files = split_files_by_random_sessions(all_files)

# --- ALIGNED PREPROCESSING ENGINE ---
def load_continuous_simulated_stream(file_list):
    """
    Simulates the ICL-MQ benchmark style by stitching separate file segments 
    into a continuous chronological stream to generate an accurate global timeline.
    """
    all_features = []
    all_labels = []
    
    window_size = 40
    step = 20
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    # Group file paths by session so we stich files together group-by-group
    session_dict = {}
    for f in file_list:
        if not os.path.exists(f): continue
        sess = os.path.basename(f).replace(".csv", "").split("_")[-1]
        if sess not in session_dict: session_dict[sess] = []
        session_dict[sess].append(f)
        
    for sess_id, paths in session_dict.items():
        # Combine the segment chunks to build a single continuous tracking dataframe
        session_df_list = []
        for p in paths:
            df = pd.read_csv(p)
            base_name = os.path.basename(p).replace(".csv", "")
            df['target_zone'] = "_".join(base_name.split("_")[:-1])
            session_df_list.append(df)
            
        # This combined matrix mimics a continuous 2-minute sensor log
        continuous_session_df = pd.concat(session_df_list, ignore_index=True)
        
        # Apply the smoothing filter across the entire continuous run
        df_copy = continuous_session_df[sensor_cols].copy()
        for col in sensor_cols:
            df_copy[col] = df_copy[col].rolling(window=5, center=True, min_periods=1).mean()
            df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)
            
        total_session_rows = len(df_copy)
        
        # Extract sliding windows over the continuous session stream
        for i in range(0, total_session_rows - window_size, step):
            window = df_copy.iloc[i : i + window_size]
            
            # --- THE BREAKTHROUGH FIX ---
            # Measures where the window sits relative to the entire session, matching the ICL notebook style
            global_relative_position = (i + window_size // 2) / total_session_rows
            features = [global_relative_position]
            
            # Statistical variances
            for col in sensor_cols:
                features.extend([window[col].std(), window[col].max() - window[col].min()])
                
            # Delta shifts
            for col in sensor_cols:
                delta_col = f'{col}_delta'
                features.extend([window[delta_col].mean(), window[delta_col].std(), window[delta_col].max() - window[delta_col].min()])
            
            # Total Magnitudes
            mag_acc = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
            mag_gyro = np.sqrt(window['gx']**2 + window['gy']**2 + window['gz']**2)
            features.extend([mag_acc.std(), mag_acc.max() - mag_acc.min(), mag_gyro.std(), mag_gyro.max() - mag_gyro.min()])
            
            # Resolve the window label by finding the most common class within those 40 rows
            window_labels = continuous_session_df['target_zone'].iloc[i : i + window_size]
            majority_label = Counter(window_labels).most_common(1)[0][0]
            
            all_features.append(features)
            all_labels.append(majority_label)
            
    return np.array(all_features), np.array(all_labels)

# Extract features using the combined global streams
X_train, y_train = load_continuous_simulated_stream(train_files)
X_test, y_test = load_continuous_simulated_stream(test_files)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate Raw Un-smoothed predictions
print("--- Evaluation on Unseen Test Session ---")
raw_predictions = model.predict(X_test)
print(classification_report(y_test, raw_predictions))

# Output-level Prediction Smoothing
smoothed_predictions = []
filter_size = 7
for i in range(len(raw_predictions)):
    start_idx = max(0, i - filter_size // 2)
    end_idx = min(len(raw_predictions), i + filter_size // 2 + 1)
    neighborhood = raw_predictions[start_idx:end_idx]
    majority_vote = Counter(neighborhood).most_common(1)[0][0]
    smoothed_predictions.append(majority_vote)

print("--- Evaluation AFTER Prediction Smoothing ---")
print(classification_report(y_test, smoothed_predictions))