import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score

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
    
    window_size = 20
    step = 10
    
    for f in file_list:
        # Load single file
        df = pd.read_csv(f)
        sensor_data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
        
        # Get clean label from filename (e.g., "upper_front_buccal_s1" -> "upper_front_buccal")
        base_name = os.path.basename(f).replace(".csv", "")
        clean_label = "_".join(base_name.split("_")[:-1]) 
        
        # Slide window strictly WITHIN this single file
        for i in range(0, len(sensor_data) - window_size, step):
            window = sensor_data.iloc[i : i + window_size]
            
            features = []
            for col in window.columns:
                features.extend([
                    window[col].mean(), 
                    window[col].std(), 
                    window[col].max(), 
                    window[col].min()
                ])
            
            # Scale-invariant feature: Magnitude calculation to handle slight grip shifts
            mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
            features.extend([mag.mean(), mag.std()])
            
            all_features.append(features)
            all_labels.append(clean_label)
            
    return np.array(all_features), np.array(all_labels)

X_train, y_train = load_and_label_by_file(train_files)
X_test, y_test = load_and_label_by_file(test_files)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on Session 3
print("--- Evaluation on Unseen Session 3 ---")
print(classification_report(y_test, model.predict(X_test)))

# 1. Get predictions for your unseen test files (Session 4)
y_pred = model.predict(X_test)

# 2. Compute the raw confusion matrix values
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# 3. Plot the confusion matrix cleanly
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

# Plot options: choose a clean blue color scheme and rotate text for readability
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix: Cross-Session Generalization Evaluation")
plt.tight_layout()
plt.show()