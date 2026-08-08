import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import random
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import json

# NEW: Import the local Ollama SDK
import ollama 

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================
# all_files = [
#     "../data/3/upper_front_buccal_s1.csv", "../data/3/upper_front_lingual_s1.csv",
#     "../data/3/upper_left_buccal_s1.csv", "../data/3/upper_left_lingual_s1.csv",
#     "../data/3/upper_left_occlusal_s1.csv", "../data/3/upper_right_buccal_s1.csv",
#     "../data/3/upper_right_lingual_s1.csv", "../data/3/upper_right_occlusal_s1.csv",
    
#     "../data/3/upper_front_buccal_s2.csv", "../data/3/upper_front_lingual_s2.csv",
#     "../data/3/upper_left_buccal_s2.csv", "../data/3/upper_left_lingual_s2.csv",
#     "../data/3/upper_left_occlusal_s2.csv", "../data/3/upper_right_buccal_s2.csv",
#     "../data/3/upper_right_lingual_s2.csv", "../data/3/upper_right_occlusal_s2.csv",

#     "../data/3/upper_front_buccal_s3.csv", "../data/3/upper_front_lingual_s3.csv",
#     "../data/3/upper_left_buccal_s3.csv", "../data/3/upper_left_lingual_s3.csv",
#     "../data/3/upper_left_occlusal_s3.csv", "../data/3/upper_right_buccal_s3.csv",
#     "../data/3/upper_right_lingual_s3.csv", "../data/3/upper_right_occlusal_s3.csv",
    
#     "../data/3/upper_front_buccal_s4.csv", "../data/3/upper_front_lingual_s4.csv",
#     "../data/3/upper_left_buccal_s4.csv", "../data/3/upper_left_lingual_s4.csv",
#     "../data/3/upper_left_occlusal_s4.csv", "../data/3/upper_right_buccal_s4.csv",
#     "../data/3/upper_right_lingual_s4.csv", "../data/3/upper_right_occlusal_s4.csv"
# ]
all_files = [
    # Session 1 (Chronological)
    "../data/3/upper_front_buccal_s1.csv", 
    "../data/3/upper_left_buccal_s1.csv",
    "../data/3/upper_right_buccal_s1.csv",
    "../data/3/upper_left_occlusal_s1.csv",
    "../data/3/upper_right_occlusal_s1.csv",
    "../data/3/upper_left_lingual_s1.csv",
    "../data/3/upper_front_lingual_s1.csv",
    "../data/3/upper_right_lingual_s1.csv",
    
    # Session 2 (Chronological)
    "../data/3/upper_front_buccal_s2.csv", 
    "../data/3/upper_left_buccal_s2.csv",
    "../data/3/upper_right_buccal_s2.csv",
    "../data/3/upper_left_occlusal_s2.csv",
    "../data/3/upper_right_occlusal_s2.csv",
    "../data/3/upper_left_lingual_s2.csv",
    "../data/3/upper_front_lingual_s2.csv",
    "../data/3/upper_right_lingual_s2.csv",

    # Session 3 (Chronological)
    "../data/3/upper_front_buccal_s3.csv", 
    "../data/3/upper_left_buccal_s3.csv",
    "../data/3/upper_right_buccal_s3.csv",
    "../data/3/upper_left_occlusal_s3.csv",
    "../data/3/upper_right_occlusal_s3.csv",
    "../data/3/upper_left_lingual_s3.csv",
    "../data/3/upper_front_lingual_s3.csv",
    "../data/3/upper_right_lingual_s3.csv",
    
    # Session 4 (Chronological)
    "../data/3/upper_front_buccal_s4.csv", 
    "../data/3/upper_left_buccal_s4.csv",
    "../data/3/upper_right_buccal_s4.csv",
    "../data/3/upper_left_occlusal_s4.csv",
    "../data/3/upper_right_occlusal_s4.csv",
    "../data/3/upper_left_lingual_s4.csv",
    "../data/3/upper_front_lingual_s4.csv",
    "../data/3/upper_right_lingual_s4.csv"
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

    print("Train Files:", train_sessions)
    print("Test Files:", test_sessions)

    return train_files, test_files

train_files, test_files = split_files_by_random_sessions(all_files)

# ==========================================
# 2. FEATURE EXTRACTION (Soft Gravity)
# ==========================================
def load_continuous_simulated_stream(file_list):
    all_features = []
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
            df_copy[f'{col}_delta'] = df_copy[col].diff().fillna(0)
            
        total_session_rows = len(df_copy)
        
        for i in range(0, total_session_rows - window_size, step):
            window = df_copy.iloc[i : i + window_size]
            global_relative_position = (i + window_size // 2) / total_session_rows
            features = [global_relative_position]
            
            # The Soft Gravity Fix
            for col in sensor_cols:
                features.extend([window[col].mean(), window[col].std(), window[col].max() - window[col].min()])
            for col in sensor_cols:
                delta_col = f'{col}_delta'
                features.extend([window[delta_col].mean(), window[delta_col].std(), window[delta_col].max() - window[delta_col].min()])
            
            mag_acc = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
            mag_gyro = np.sqrt(window['gx']**2 + window['gy']**2 + window['gz']**2)
            features.extend([mag_acc.std(), mag_acc.max() - mag_acc.min(), mag_gyro.std(), mag_gyro.max() - mag_gyro.min()])
            
            window_labels = continuous_session_df['target_zone'].iloc[i : i + window_size]
            majority_label = Counter(window_labels).most_common(1)[0][0]
            
            all_features.append(features)
            all_labels.append(majority_label)
            
    return np.array(all_features), np.array(all_labels)

X_train, y_train = load_continuous_simulated_stream(train_files)
X_test, y_test = load_continuous_simulated_stream(test_files)

# ==========================================
# 3. TRAIN RANDOM FOREST
# ==========================================
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


# ==========================================
# 4. NEURO-SYMBOLIC LLM AGENT PIPELINE (OLLAMA)
# ==========================================
print("\n🤖 Activating LOCAL Ollama Domain Agent...")

# Ask the RF for PROBABILITIES
rf_probabilities = model.predict_proba(X_test)
classes = model.classes_

# Analyze a 15-step slice of the data
test_slice_start = 50 
test_slice_end = 65
llm_final_predictions = []

# NEW: Create a copy of the RF predictions to inject our LLM corrections into
hybrid_predictions = model.predict(X_test).tolist()

# System Prompt detailing the Domain Knowledge
system_instruction = """
You are an expert AI spatial reasoning agent for a smart toothbrush. 
Your job is to look at the statistical output from a Random Forest model, look at the physical context, and correct any physically impossible teleportations across the mouth.
Human anatomy rule: Brushing transitions usually follow adjacent zones (e.g., buccal -> occlusal -> lingual). 
Respond ONLY with a JSON object containing "predicted_zone" and "reasoning".
"""

recent_history = [] 

# NOTE: Make sure your Ollama desktop app is running in the background before executing this!
for i in range(test_slice_start, test_slice_end):
    probs = rf_probabilities[i]
    top_2_indices = np.argsort(probs)[-2:][::-1]
    
    top_class = classes[top_2_indices[0]]
    top_prob = probs[top_2_indices[0]] * 100
    runner_up = classes[top_2_indices[1]]
    runner_up_prob = probs[top_2_indices[1]] * 100
    
    grav_x = round(X_test[i][1], 2)
    grav_y = round(X_test[i][4], 2)
    grav_z = round(X_test[i][7], 2)

    # Translate to Domain Language
    domain_prompt = f"""
    --- DATA FOR CURRENT WINDOW ---
    Recent History (Last 3 zones): {recent_history[-3:] if recent_history else 'Start of session'}
    Gravity Vector (Wrist Angle): X={grav_x}, Y={grav_y}, Z={grav_z}
    
    Random Forest Analysis:
    1. {top_class} ({top_prob:.1f}%)
    2. {runner_up} ({runner_up_prob:.1f}%)
    
    Task: What is the true zone? If the RF's top choice breaks physical continuity (teleportation), override it with the runner up or a logical adjacent zone.
    """
    
    try:
        # Call the local model
        response = ollama.chat(
            model='llama3.2', # Make sure to run `ollama pull llama3.2` in your terminal first!
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': domain_prompt}
            ],
            format='json' # Ollama natively forces the model to output strict JSON
        )
        
        # Parse the JSON response
        agent_decision = json.loads(response['message']['content'])
        final_zone = agent_decision.get("predicted_zone", top_class)
        reason = agent_decision.get("reasoning", "No reasoning provided.")
        
    except Exception as e:
        # Fallback if Ollama fails (e.g., server isn't running or model isn't downloaded)
        final_zone = top_class
        reason = f"Ollama Error: {str(e)}. Falling back to RF."

        
    print(f"\nStep {i} | Ground Truth: {y_test[i]}")
    print(f"RF Suggested: {top_class} ({top_prob:.1f}%)")
    print(f"🤖 LLM Decided: {final_zone}")
    print(f"   ↳ Reason: {reason}")
    
    llm_final_predictions.append(final_zone)
    recent_history.append(final_zone)
    
    # NEW: Inject the LLM's decision back into our full prediction array
    hybrid_predictions[i] = final_zone

print("\n✅ Ollama Agent evaluation complete.")

# ==========================================
# 5. LLM CONFUSION MATRIX VISUALIZATION
# ==========================================
print("\n--- Final Hybrid Evaluation (RF + LLM Overrides) ---")
print(classification_report(y_test, hybrid_predictions))

# Compute the Confusion Matrix
cm = confusion_matrix(y_test, hybrid_predictions, labels=model.classes_)

# Create and Plot the Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

plt.title("Confusion Matrix (RF + Local LLM Agent)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()