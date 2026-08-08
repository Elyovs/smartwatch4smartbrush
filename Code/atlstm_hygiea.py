import pandas as pd
import numpy as np
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. FILE CONFIGURATION (Same as before)
# ==========================================
all_files = [
    # Session 1
    "../data/3/upper_front_buccal_s1.csv", "../data/3/upper_front_lingual_s1.csv",
    "../data/3/upper_left_buccal_s1.csv", "../data/3/upper_left_lingual_s1.csv",
    "../data/3/upper_left_occlusal_s1.csv", "../data/3/upper_right_buccal_s1.csv",
    "../data/3/upper_right_lingual_s1.csv", "../data/3/upper_right_occlusal_s1.csv",
    # Session 2
    "../data/3/upper_front_buccal_s2.csv", "../data/3/upper_front_lingual_s2.csv",
    "../data/3/upper_left_buccal_s2.csv", "../data/3/upper_left_lingual_s2.csv",
    "../data/3/upper_left_occlusal_s2.csv", "../data/3/upper_right_buccal_s2.csv",
    "../data/3/upper_right_lingual_s2.csv", "../data/3/upper_right_occlusal_s2.csv",
    # Session 3
    "../data/3/upper_front_buccal_s3.csv", "../data/3/upper_front_lingual_s3.csv",
    "../data/3/upper_left_buccal_s3.csv", "../data/3/upper_left_lingual_s3.csv",
    "../data/3/upper_left_occlusal_s3.csv", "../data/3/upper_right_buccal_s3.csv",
    "../data/3/upper_right_lingual_s3.csv", "../data/3/upper_right_occlusal_s3.csv",
    # Session 4
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
        if session_id not in session_groups: session_groups[session_id] = []
        session_groups[session_id].append(f)
        
    unique_sessions = list(session_groups.keys())
    random.shuffle(unique_sessions)
    
    train_sessions = unique_sessions[:3]
    test_sessions = unique_sessions[3:]
    
    train_files = [f for sess in train_sessions for f in session_groups[sess]]
    test_files = [f for sess in test_sessions for f in session_groups[sess]]
        
    print(f"🎲 Sessions -> Train (3): {train_sessions} | Test (1): {test_sessions}\n")
    return train_files, test_files

train_files, test_files = split_files_by_random_sessions(all_files)

# ==========================================
# 2. SEQUENTIAL PREPROCESSING (Same as before)
# ==========================================
def load_sequential_stream(file_list):
    all_sequences = []
    all_labels = []
    
    window_size = 40
    step = 20
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    session_dict = {}
    for f in file_list:
        if not os.path.exists(f): continue
        sess = os.path.basename(f).replace(".csv", "").split("_")[-1]
        if sess not in session_dict: session_dict[sess] = []
        session_dict[sess].append(f)
        
    for sess_id, paths in session_dict.items():
        session_df_list = []
        for p in paths:
            df = pd.read_csv(p)
            base_name = os.path.basename(p).replace(".csv", "")
            df['target_zone'] = "_".join(base_name.split("_")[:-1])
            session_df_list.append(df)
            
        if not session_df_list: continue
        continuous_session_df = pd.concat(session_df_list, ignore_index=True)
        
        df_copy = continuous_session_df[sensor_cols].copy()
        for col in sensor_cols:
            df_copy[col] = df_copy[col].rolling(window=5, center=True, min_periods=1).mean()
            
        total_rows = len(df_copy)
        
        for i in range(0, total_rows - window_size, step):
            window_data = df_copy.iloc[i : i + window_size].values
            window_labels = continuous_session_df['target_zone'].iloc[i : i + window_size]
            majority_label = Counter(window_labels).most_common(1)[0][0]
            
            all_sequences.append(window_data)
            all_labels.append(majority_label)
            
    return np.array(all_sequences), np.array(all_labels)

X_train_raw, y_train_raw = load_sequential_stream(train_files)
X_test_raw, y_test_raw = load_sequential_stream(test_files)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_test_encoded = label_encoder.transform(y_test_raw)

X_train = torch.tensor(X_train_raw, dtype=torch.float32)
y_train = torch.tensor(y_train_encoded, dtype=torch.long)
X_test = torch.tensor(X_test_raw, dtype=torch.float32)
y_test = torch.tensor(y_test_encoded, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# ==========================================
# 3. UPDATED ARCHITECTURE (DROPOUT ADDED)
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        attn_scores = self.attention_net(lstm_outputs) 
        attn_weights = F.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1) 
        return context_vector, attn_weights

class AT_LSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_classes=8, dropout_rate=0.4): # Added dropout rate
        super(AT_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # NEW: Dropout Layer
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        
        # NEW: Apply dropout to the context vector before classifying
        context_dropped = self.dropout(context)
        out = self.classifier(context_dropped)
        
        return out

# ==========================================
# 4. UPDATED TRAINING LOOP (NOISE & WEIGHT DECAY)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Training on: {device}")

# NEW: Reduced hidden_dim from 64 to 32
model = AT_LSTM(input_dim=6, hidden_dim=32, num_classes=len(label_encoder.classes_), dropout_rate=0.4).to(device)
criterion = nn.CrossEntropyLoss()

# NEW: Added weight_decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# NEW: Data Augmentation Function (Gaussian Noise)
def add_sensor_noise(batch_x, noise_std=0.02):
    """Injects slight random variations to simulate sensor imperfection."""
    noise = torch.randn_like(batch_x) * noise_std
    return batch_x + noise

epochs = 30

print("🚀 Starting AT-LSTM Training with Anti-Overfitting Measures...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # NEW: Apply noise augmentation ONLY during training
        batch_X_noisy = add_sensor_noise(batch_X, noise_std=0.03) 
        
        optimizer.zero_grad()
        outputs = model(batch_X_noisy) # Feed the noisy data to the model
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Acc: {100 * correct / total:.2f}%")

# # ==========================================
# # 5. EVALUATION (Same as before)
# # ==========================================
# model.eval() # This automatically turns OFF the dropout layer for testing!
# all_preds = []
# all_targets = []

# with torch.no_grad():
#     for batch_X, batch_y in test_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         # Note: We do NOT add noise during evaluation
#         outputs = model(batch_X)
#         _, predicted = torch.max(outputs.data, 1)
#         all_preds.extend(predicted.cpu().numpy())
#         all_targets.extend(batch_y.cpu().numpy())

# predicted_labels = label_encoder.inverse_transform(all_preds)
# target_labels = label_encoder.inverse_transform(all_targets)

# print("\n--- AT-LSTM Evaluation on Unseen Test Session ---")
# print(classification_report(target_labels, predicted_labels))

# cm = confusion_matrix(target_labels, predicted_labels, labels=label_encoder.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# fig, ax = plt.subplots(figsize=(10, 8))
# disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
# plt.title("Hygiea+ AT-LSTM Confusion Matrix (Regularized)", fontsize=14, pad=20)
# plt.tight_layout()
# plt.show()

# ==========================================
# 5. EVALUATION & OUTPUT SMOOTHING (with filter)
# ==========================================
model.eval() 
all_preds = []
all_targets = []

# 1. Get raw predictions from the AT-LSTM
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

# 2. Apply the Output-Level Smoothing Filter (Majority Vote)
smoothed_preds = []
filter_size = 7
for i in range(len(all_preds)):
    start_idx = max(0, i - filter_size // 2)
    end_idx = min(len(all_preds), i + filter_size // 2 + 1)
    neighborhood = all_preds[start_idx:end_idx]
    majority_vote = Counter(neighborhood).most_common(1)[0][0]
    smoothed_preds.append(majority_vote)

# 3. Decode integers back to string labels using the SMOOTHED predictions
predicted_labels = label_encoder.inverse_transform(smoothed_preds)
target_labels = label_encoder.inverse_transform(all_targets)

print("\n--- AT-LSTM Evaluation (AFTER 7-Step Output Smoothing) ---")
print(classification_report(target_labels, predicted_labels))

# 4. Confusion Matrix Visualization
cm = confusion_matrix(target_labels, predicted_labels, labels=label_encoder.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
plt.title("Hygiea+ AT-LSTM Confusion Matrix (Smoothed)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()