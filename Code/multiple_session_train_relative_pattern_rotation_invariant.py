import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load and Label Data
train_files = [
    "../data/2/upper_front_buccal_s1.csv",
    "../data/2/upper_front_lingual_s1.csv",
    "../data/2/upper_left_buccal_s1.csv",
    "../data/2/upper_left_lingual_s1.csv",
    "../data/2/upper_left_occlusal_s1.csv",
    "../data/2/upper_right_buccal_s1.csv",
    "../data/2/upper_right_lingual_s1.csv",
    "../data/2/upper_right_occlusal_s1.csv",
    "../data/2/upper_front_buccal_s2.csv",
    "../data/2/upper_front_lingual_s2.csv",
    "../data/2/upper_left_buccal_s2.csv",
    "../data/2/upper_left_lingual_s2.csv",
    "../data/2/upper_left_occlusal_s2.csv",
    "../data/2/upper_right_buccal_s2.csv",
    "../data/2/upper_right_lingual_s2.csv",
    "../data/2/upper_right_occlusal_s2.csv"
]

test_files = [
    "../data/2/upper_front_buccal_s4.csv",
    "../data/2/upper_front_lingual_s4.csv",
    "../data/2/upper_left_buccal_s4.csv",
    "../data/2/upper_left_lingual_s4.csv",
    "../data/2/upper_left_occlusal_s4.csv",
    "../data/2/upper_right_buccal_s4.csv",
    "../data/2/upper_right_lingual_s4.csv",
    "../data/2/upper_right_occlusal_s4.csv"
]

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

# Confusion Matrix Visualizing
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix: Pattern-Based Cross-Session Generalization")
plt.tight_layout()
plt.show()