import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score

# 1. Load and Label Data
files = [
            "../data/1/upper_front_buccal.csv",
            "../data/1/upper_front_lingual.csv",
            "../data/1/upper_left_buccal.csv",
            "../data/1/upper_left_lingual.csv",
            "../data/1/upper_left_occlusal.csv",
            "../data/1/upper_right_buccal.csv",
            "../data/1/upper_right_lingual.csv",
            "../data/1/upper_right_occlusal.csv"
        ]
all_data = pd.concat([pd.read_csv(f).assign(label=f.replace(".csv","")) for f in files])

# 2. Feature Extraction (Windowing)
def extract_features(df, window_size=20):
    X, y = [], []
    for label in df['label'].unique():
        temp = df[df['label'] == label].drop('label', axis=1)
        for i in range(0, len(temp) - window_size, 10): # 10-step overlap
            window = temp.iloc[i:i+window_size]
            features = []
            for col in window.columns:
                features.extend([window[col].mean(), window[col].std(), window[col].max(), window[col].min()])
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Example Usage:
# result = test_new_file("new_test_data.csv", model)
def test_new_file(file_path, model, window_size=20, step=10):
    """
    Inputs a new CSV file and outputs the model's classification.
    """
    # 1. Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading file: {e}"

    # Ensure we only use the 6 sensor columns
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    if not all(col in df.columns for col in sensor_cols):
        return "Error: File missing required sensor columns (ax, ay, az, gx, gy, gz)"
    
    data = df[sensor_cols]
    windowed_predictions = []
    
    # 2. Process windows
    for i in range(0, len(data) - window_size, step):
        window = data.iloc[i : i + window_size]
        
        # 3. Feature Extraction (Must match training exactly)
        features = []
        for col in sensor_cols:
            features.extend([
                window[col].mean(), 
                window[col].std(), 
                window[col].max(), 
                window[col].min()
            ])
        
        # 4. Predict
        pred = model.predict(np.array(features).reshape(1, -1))
        windowed_predictions.append(pred[0])
    
    if not windowed_predictions:
        return "File too short for the window size."

    # 5. Summarize the whole session
    counts = Counter(windowed_predictions)
    most_common, num_votes = counts.most_common(1)[0]
    confidence = (num_votes / len(windowed_predictions)) * 100

    print(f"--- Analysis for {file_path} ---")
    print(f"Total Windows Analyzed: {len(windowed_predictions)}")
    print(f"Most Frequent Prediction: {most_common}")
    print(f"Confidence (Majority Vote): {confidence:.2f}%")
    print(f"Full Vote Distribution: {dict(counts)}")
    
    return most_common

X, y = extract_features(all_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
print(classification_report(y_test, model.predict(X_test)))

# 1. Predict labels for the windowed test set
y_pred_w = model.predict(X_test)

# 2. Create the confusion matrix object
cm = confusion_matrix(y_test, y_pred_w, labels=model.classes_)

# Calculate predictions for both sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate Accuracy
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# test using different test data
result = test_new_file("../test_set1_right_occlusal.csv", model)
print("test classification result: " + str(result))

# 3. Plot the matrix
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

# Customize the look (colors, label rotation)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix: Windowed Classification")
plt.tight_layout()
plt.show()