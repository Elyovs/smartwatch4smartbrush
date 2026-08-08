import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.metrics import classification_report

# 1. Group files cleanly by their natural sessions
session1_files = [
    "../data/2/upper_front_buccal_s1.csv", "../data/2/upper_front_lingual_s1.csv",
    "../data/2/upper_left_buccal_s1.csv", "../data/2/upper_left_lingual_s1.csv",
    "../data/2/upper_left_occlusal_s1.csv", "../data/2/upper_right_buccal_s1.csv",
    "../data/2/upper_right_lingual_s1.csv", "../data/2/upper_right_occlusal_s1.csv"
]

session2_files = [
    "../data/2/upper_front_buccal_s2.csv", "../data/2/upper_front_lingual_s2.csv",
    "../data/2/upper_left_buccal_s2.csv", "../data/2/upper_left_lingual_s2.csv",
    "../data/2/upper_left_occlusal_s2.csv", "../data/2/upper_right_buccal_s2.csv",
    "../data/2/upper_right_lingual_s2.csv", "../data/2/upper_right_occlusal_s2.csv"
]

session4_files = [
    "../data/2/upper_front_buccal_s4.csv", "../data/2/upper_front_lingual_s4.csv",
    "../data/2/upper_left_buccal_s4.csv", "../data/2/upper_left_lingual_s4.csv",
    "../data/2/upper_left_occlusal_s4.csv", "../data/2/upper_right_buccal_s4.csv",
    "../data/2/upper_right_lingual_s4.csv", "../data/2/upper_right_occlusal_s4.csv"
]

# 2. Extract Raw Windows and Apply Standard Scaling per window
def load_normalized_windows(file_list, window_size=40, step=20):
    all_X, all_y = [], []
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    scaler = StandardScaler()
    
    for f in file_list:
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        df_copy = df[sensor_cols].copy()
        
        base_name = os.path.basename(f).replace(".csv", "")
        clean_label = "_".join(base_name.split("_")[:-1]) 
        
        for i in range(0, len(df_copy) - window_size, step):
            window_matrix = df_copy.iloc[i : i + window_size].values
            
            # --- IMPROVEMENT A: STANDARDIZE SCALES ---
            # Zero-mean and unit-variance normalization transforms raw magnitudes 
            # into relative wave shapes that are identical across sessions
            window_normalized = scaler.fit_transform(window_matrix)
            
            all_X.append(window_normalized)
            all_y.append(clean_label)
            
    return np.array(all_X), np.array(all_y)

print("Loading and normalizing sessions independently...")
X_s1, y_s1 = load_normalized_windows(session1_files)
X_s2, y_s2 = load_normalized_windows(session2_files)
X_s4, y_s4 = load_normalized_windows(session4_files)

# Combine labels to fit the global encoder mapping
all_labels = np.concatenate([y_s1, y_s2, y_s4])
encoder = LabelEncoder()
encoder.fit(all_labels)

y_train = encoder.transform(y_s1)
y_val = encoder.transform(y_s2)
y_test = encoder.transform(y_s4)

# 3. Setup explicitly segregated datasets
X_train, X_val, X_test = X_s1, X_s2, X_s4
print(f"Train matrices: {X_train.shape} | Val matrices: {X_val.shape} | Test matrices: {X_test.shape}")

# 4. Build a regularized 1D CNN Architecture
# Added higher dropout and weight regularizers to aggressively combat overfitting
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(40, 6)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),  # Increased to break noise reliance
    
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),  # Increased to prevent filter memorization
    
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train using True Cross-Session Validation Data
print("\nTraining Deep Learning 1D CNN with Clean Session Validation...")
model.fit(
    X_train, y_train, 
    epochs=40, 
    batch_size=32, 
    validation_data=(X_val, y_val)  # --- IMPROVEMENT B: SEGREGATED VALIDATION ---
)

# 6. Final Evaluation on Session 4
print("\n--- 1D CNN Evaluation on Unseen Session 4 ---")
y_pred_probs = model.predict(X_test)
y_pred_encoded = np.argmax(y_pred_probs, axis=1)
y_pred_strings = encoder.inverse_transform(y_pred_encoded)
y_test_strings = encoder.inverse_transform(y_test)

print(classification_report(y_test_strings, y_pred_strings))