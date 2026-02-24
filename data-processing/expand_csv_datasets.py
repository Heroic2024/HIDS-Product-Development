import csv
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Paths
features_dir = os.path.join(os.path.dirname(__file__), '..', 'hids_dataset', 'features')
train_path = os.path.join(features_dir, 'train.csv')
test_path = os.path.join(features_dir, 'test.csv')

def load_csv_data(filepath):
    """Load CSV and return headers and data rows"""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows

def rows_to_array(rows, headers):
    """Convert rows to numpy array (excluding label)"""
    feature_headers = [h for h in headers if h != 'label']
    data = []
    labels = []
    for row in rows:
        features = [float(row[h]) for h in feature_headers]
        label = int(row['label'])
        data.append(features)
        labels.append(label)
    return np.array(data), np.array(labels), feature_headers

def generate_synthetic_rows(features, labels, feature_headers, num_rows):
    """Generate synthetic rows based on feature distribution"""
    num_features = features.shape[1]
    
    # Calculate statistics for each feature from existing data
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    
    # Get class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    total_original = len(labels)
    
    # Generate proportional classes
    benign_count = int(num_rows * (class_dist.get(1, 1) / total_original))
    malicious_count = num_rows - benign_count
    
    synthetic_rows = []
    synthetic_labels = []
    
    # Generate benign samples (label=1)
    for _ in range(benign_count):
        row = np.random.normal(feature_means, feature_stds + 0.1)  # Added small noise
        synthetic_rows.append(row)
        synthetic_labels.append(1)
    
    # Generate malicious samples (label=0)
    for _ in range(malicious_count):
        row = np.random.normal(feature_means, feature_stds + 0.1)
        synthetic_rows.append(row)
        synthetic_labels.append(0)
    
    return np.array(synthetic_rows), np.array(synthetic_labels)

def save_expanded_csv(filepath, original_rows, headers, synthetic_features, synthetic_labels, feature_headers):
    """Write expanded CSV with original + synthetic rows"""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        # Write original rows
        for row in original_rows:
            writer.writerow(row)
        
        # Write synthetic rows
        for features, label in zip(synthetic_features, synthetic_labels):
            row_dict = {h: str(val) for h, val in zip(feature_headers, features)}
            row_dict['label'] = str(label)
            writer.writerow(row_dict)

# Load existing data
print("[*] Loading existing train.csv...")
train_headers, train_rows = load_csv_data(train_path)
train_features, train_labels, train_feature_headers = rows_to_array(train_rows, train_headers)
print(f"    Original train rows: {len(train_rows)}")

print("[*] Loading existing test.csv...")
test_headers, test_rows = load_csv_data(test_path)
test_features, test_labels, test_feature_headers = rows_to_array(test_rows, test_headers)
print(f"    Original test rows: {len(test_rows)}")

# Generate synthetic data
print("\n[*] Generating 5000 synthetic rows for training data...")
train_synthetic_features, train_synthetic_labels = generate_synthetic_rows(
    train_features, train_labels, train_feature_headers, 5000
)
print(f"    Generated: {len(train_synthetic_labels)} rows")
print(f"    Class distribution: {np.unique(train_synthetic_labels, return_counts=True)}")

print("[*] Generating 5000 synthetic rows for test data...")
test_synthetic_features, test_synthetic_labels = generate_synthetic_rows(
    test_features, test_labels, test_feature_headers, 5000
)
print(f"    Generated: {len(test_synthetic_labels)} rows")
print(f"    Class distribution: {np.unique(test_synthetic_labels, return_counts=True)}")

# Save expanded datasets
print("\n[*] Saving expanded train.csv...")
save_expanded_csv(train_path, train_rows, train_headers, train_synthetic_features, train_synthetic_labels, train_feature_headers)
new_train_count = len(train_rows) + len(train_synthetic_labels)
print(f"    ✓ Saved train.csv ({new_train_count} rows)")

print("[*] Saving expanded test.csv...")
save_expanded_csv(test_path, test_rows, test_headers, test_synthetic_features, test_synthetic_labels, test_feature_headers)
new_test_count = len(test_rows) + len(test_synthetic_labels)
print(f"    ✓ Saved test.csv ({new_test_count} rows)")

print("\n[+] Dataset expansion complete!")
print(f"    train.csv: {len(train_rows)} → {new_train_count} rows (+5000)")
print(f"    test.csv: {len(test_rows)} → {new_test_count} rows (+5000)")
