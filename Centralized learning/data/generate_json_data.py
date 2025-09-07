import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import json
import os

# Parameters
PERCENT_DATA = 0.12
N_FOLDS = 5
N_TIME_STEPS = 30
STEP = 30  # No overlap

# --- 1. Load the full dataset ---
print("--- 1. Loading Full Dataset ---")
try:
    df_har = pd.read_csv("WISDM_ar_v1.1_raw.txt", header=None,
                         names=["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"])
    # Clean the 'z-axis' column and handle potential errors
    df_har["z-axis"] = df_har["z-axis"].astype(str).str.replace(";", "", regex=False).astype(float)
    df_har.dropna(axis=0, how='any', inplace=True)
    df_har = df_har[df_har["timestamp"] != 0]

    print(f"✅ Full dataset loaded successfully.")
    print(f"  - Initial shape: {df_har.shape}")
    print(f"  - Total unique users: {df_har['user'].nunique()}")
    print("  - Activity distribution in the full dataset:")
    print(df_har['activity'].value_counts().to_string())

except FileNotFoundError:
    print("❌ ERROR: 'WISDM_ar_v1.1_raw.txt' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2 & 3. Create a subset based on the first 12% of users ---
print(f"\n--- 2 & 3. Subsetting Data ---")
print(f"Identifying users from the first {PERCENT_DATA * 100:.0f}% of raw data entries...")
rows_to_keep = int(len(df_har) * PERCENT_DATA)
users_in_subset = df_har.iloc[:rows_to_keep]["user"].unique()
print(f"  - Found {len(users_in_subset)} unique users in the initial slice.")

print("Creating subset with all data from these selected users...")
df_subset = df_har[df_har["user"].isin(users_in_subset)].reset_index(drop=True)
subset_percentage = (len(df_subset) / len(df_har)) * 100
print(f"✅ Subset created.")
print(f"  - Shape of the subsetted data: {df_subset.shape}")
print(f"  - The subset contains {len(df_subset)} rows ({subset_percentage:.2f}% of the total data).")

# --- 4. Encode activity labels ---
print("\n--- 4. Encoding Activity Labels ---")
le = LabelEncoder()
df_subset["label"] = le.fit_transform(df_subset["activity"].values.ravel())

print("✅ Labels encoded.")
print("  - Activity to Label Mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"    {class_name} -> {i}")

# --- 5. Segment data into windows ---
print("\n--- 5. Segmenting Data into Windows ---")
X, y = [], []
for i in range(0, len(df_subset) - N_TIME_STEPS, STEP):
    xs = df_subset["x-axis"].values[i: i + N_TIME_STEPS]
    ys = df_subset["y-axis"].values[i: i + N_TIME_STEPS]
    zs = df_subset["z-axis"].values[i: i + N_TIME_STEPS]

    # The label for a window is the most frequent activity within it
    label = stats.mode(df_subset["label"].values[i: i + N_TIME_STEPS])[0][0]

    X.append([xs, ys, zs])
    y.append(label)

# Reshape data to (samples, timesteps, features)
X = np.transpose(np.array(X), (0, 2, 1))
y = np.array(y)
print("✅ Data windowing complete.")
print(f"  - Total windows (samples) created: {len(X)}")
print(f"  - Shape of feature matrix X: {X.shape}")
print(f"  - Shape of label vector y: {y.shape}")

# --- 6. Create output folders ---
print("\n--- 6. Creating Output Directories ---")
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
print("  - Directories 'data/train' and 'data/test' are ready.")

# --- 7. Split into folds, normalize, and save ---
print(f"\n--- 7. Creating and Saving {N_FOLDS} Folds ---")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\nProcessing Fold {fold_idx}/{N_FOLDS}...")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  - Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Normalize using training stats for the current fold
    mean_vals = X_train.mean(axis=(0, 1))
    std_vals = X_train.std(axis=(0, 1))
    X_train = (X_train - mean_vals) / (std_vals + 1e-8)  # Add epsilon for stability
    X_test = (X_test - mean_vals) / (std_vals + 1e-8)

    print(f"  - Normalization stats (mean): {np.round(mean_vals, 3)}")
    print(f"  - Normalization stats (std):  {np.round(std_vals, 3)}")

    # Prepare data in the specified JSON format
    train_data = {"users": ["merged_user"],
                  "user_data": {"merged_user": {"x": X_train.tolist(), "y": y_train.tolist()}},
                  "num_samples": {"merged_user": len(y_train)}}
    test_data = {"users": ["merged_user"], "user_data": {"merged_user": {"x": X_test.tolist(), "y": y_test.tolist()}},
                 "num_samples": {"merged_user": len(y_test)}}

    # Save to JSON files
    train_path = f"train/fold_{fold_idx}_train.json"
    test_path = f"test/fold_{fold_idx}_test.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f)
    with open(test_path, "w") as f:
        json.dump(test_data, f)

    print(f"  - Saved fold data to '{train_path}' and '{test_path}'")

print(f"\n✅ All {N_FOLDS} folds have been generated successfully.")