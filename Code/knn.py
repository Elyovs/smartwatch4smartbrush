import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from collections import Counter

train_files = [
    "../data/2/upper_front_buccal_s1.csv", "../data/2/upper_front_lingual_s1.csv",
    "../data/2/upper_left_buccal_s1.csv", "../data/2/upper_left_lingual_s1.csv",
    "../data/2/upper_left_occlusal_s1.csv", "../data/2/upper_right_buccal_s1.csv",
    "../data/2/upper_right_lingual_s1.csv", "../data/2/upper_right_occlusal_s1.csv",
    "../data/2/upper_front_buccal_s2.csv", "../data/2/upper_front_lingual_s2.csv",
    "../data/2/upper_left_buccal_s2.csv", "../data/2/upper_left_lingual_s2.csv",
    "../data/2/upper_left_occlusal_s2.csv", "../data/2/upper_right_buccal_s2.csv",
    "../data/2/upper_right_lingual_s2.csv", "../data/2/upper_right_occlusal_s2.csv"
]

test_files = [
    "../data/2/upper_front_buccal_s4.csv", "../data/2/upper_front_lingual_s4.csv",
    "../data/2/upper_left_buccal_s4.csv", "../data/2/upper_left_lingual_s4.csv",
    "../data/2/upper_left_occlusal_s4.csv", "../data/2/upper_right_buccal_s4.csv",
    "../data/2/upper_right_lingual_s4.csv", "../data/2/upper_right_occlusal_s4.csv"
]

def load_order_independent_data(file_list):
    all_features = []
    all_labels = []
    
    window_size = 40
    step = 20
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for f in file_list:
        df = pd.read_csv(f)
        
        # --- FEATURE ADVANCEMENT 1: STABLE ANGULAR POSTURE ---
        # Calculate pitch and roll natively to track the 3D tilt of the handle
        df['pitch'] = np.arctan2(df['ax'], np.sqrt(df['ay']**2 + df['az']**2))
        df['roll'] = np.arctan2(df['ay'], df['az'])
        
        df_copy = df[sensor_cols + ['pitch', 'roll']].copy()
        for col in sensor_cols:
            df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)
            
        base_name = os.path.basename(f).replace(".csv", "")
        clean_label = "_".join(base_name.split("_")[:-1]) 
        
        for i in range(0, len(df_copy) - window_size, step):
            window = df_copy.iloc[i : i + window_size]
            
            # NO ORDER INDEX HERE - COMPLETELY DECOUPLED FROM TIME
            features = []
            
            # Extract Pitch and Roll orientation anchors
            features.extend([
                window['pitch'].mean(), window['pitch'].std(),
                window['roll'].mean(), window['roll'].std()
            ])
            
            # Extract standard deviations (Vibration signatures)
            for col in sensor_cols:
                # --- FEATURE ADVANCEMENT 2: INTERQUARTILE RANGE (IQR) ---
                # Replaces fragile max-min with stable percentiles
                iqr = np.percentile(window[col], 75) - np.percentile(window[col], 25)
                features.extend([window[col].std(), iqr])
                
            # Extract delta patterns (Velocity waves)
            for col in sensor_cols:
                delta_col = f'{col}_delta'
                features.extend([window[delta_col].mean(), window[delta_col].std()])
            
            # --- FEATURE ADVANCEMENT 3: CROSS-CHANNEL AXIS CORRELATION ---
            # Tracks how axes move in relation to one another based on your physical stroke angle
            acc_corr = np.corrcoef(window['ax'], window['ay'])[0, 1]
            gyro_corr = np.corrcoef(window['gx'], window['gy'])[0, 1]
            # Replace NaNs with 0 if there's absolutely no movement
            features.extend([0.0 if np.isnan(acc_corr) else acc_corr, 
                             0.0 if np.isnan(gyro_corr) else gyro_corr])
            
            # Magnitude Vectors
            mag_acc = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
            mag_gyro = np.sqrt(window['gx']**2 + window['gy']**2 + window['gz']**2)
            features.extend([mag_acc.mean(), mag_acc.std(), mag_gyro.mean(), mag_gyro.std()])
            
            all_features.append(features)
            all_labels.append(clean_label)
            
    return np.array(all_features), np.array(all_labels)

# Extract and Scale
print("Extracting order-independent kinematic signatures...")
X_train, y_train = load_order_independent_data(train_files)
X_test, y_test = load_order_independent_data(test_files)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
knn.fit(X_train_scaled, y_train)

# Raw output evaluation
raw_preds = knn.predict(X_test_scaled)

# Prediction Smoothing (Cleans up localized noise spikes sequentially)
smoothed_predictions = []
filter_size = 7
for i in range(len(raw_preds)):
    start_idx = max(0, i - filter_size // 2)
    end_idx = min(len(raw_preds), i + filter_size // 2 + 1)
    neighborhood = raw_preds[start_idx:end_idx]
    majority_vote = Counter(neighborhood).most_common(1)[0][0]
    smoothed_predictions.append(majority_vote)

print("\n--- Order-Independent KNN Performance Report ---")
print(classification_report(y_test, smoothed_predictions))