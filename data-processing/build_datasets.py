import os
import json
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_features_json(json_path):
    """Load features from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def detect_feature_columns(sample_dict):
    """
    Detect feature columns automatically.
    Exclude: label, filename, original_category (metadata and labels).
    """
    metadata_keys = {'label', 'filename', 'original_category'}
    feature_cols = [k for k in sample_dict.keys() if k not in metadata_keys]
    return sorted(feature_cols)


def extract_features_and_labels(data, feature_cols):
    """
    Extract feature matrix X and labels y from the data list.
    
    Args:
        data: List of dictionaries from JSON
        feature_cols: List of feature column names
        
    Returns:
        X: numpy array of features (n_samples, n_features)
        y: numpy array of labels
        metadata: dict with original_category and filename for each sample
    """
    X = []
    y = []
    metadata = []
    
    for sample in data:
        # Extract features
        features = [float(sample.get(col, 0)) for col in feature_cols]
        X.append(features)
        
        # Extract label
        y.append(sample.get('label', 0))
        
        # Store metadata
        metadata.append({
            'filename': sample.get('filename', 'unknown'),
            'original_category': sample.get('original_category', 'unknown')
        })
    
    return np.array(X, dtype=np.float32), np.array(y), metadata


def print_class_distribution(y, metadata):
    """Print class distribution."""
    print('\n=== Class Distribution ===')
    unique, counts = np.unique(y, return_counts=True)
    
    for label, count in zip(unique, counts):
        # Map label back to category
        categories = [m['original_category'] for m, l in zip(metadata, y) if l == label]
        if categories:
            category = Counter(categories).most_common(1)[0][0]
            print(f"Label {label} ({category}): {count} samples")
    
    print(f'Total samples: {len(y)}')
    print(f'Benign (label=1): {sum(y == 1)}')
    print(f'Malicious (label=0): {sum(y == 0)}')


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    features_dir = os.path.join(repo_root, 'hids_dataset', 'features')
    
    json_path = os.path.join(features_dir, 'dataset_features.json')
    
    if not os.path.exists(json_path):
        print(f'Error: {json_path} not found')
        return
    
    print('Loading features from JSON...')
    data = load_features_json(json_path)
    
    # Detect feature columns
    feature_cols = detect_feature_columns(data[0])
    print(f'\n[+] Detected {len(feature_cols)} feature columns')
    print(f'    Features: {feature_cols}')
    
    # Extract features and labels
    print('\nExtracting features and labels...')
    X, y, metadata = extract_features_and_labels(data, feature_cols)
    print(f'[+] Feature matrix shape: {X.shape}')
    print(f'[+] Label vector shape: {y.shape}')
    
    # Print class distribution
    print_class_distribution(y, metadata)
    
    # Initialize scaler
    print('\n=== Standardizing Features ===')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f'[+] Features standardized (mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f})')
    
    # Train-test split with reproducibility
    print('\n=== Splitting Data (80-20) ===')
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'[+] Training set: {X_train.shape[0]} samples')
    print(f'[+] Testing set: {X_test.shape[0]} samples')
    
    # Save datasets
    print('\n=== Saving Datasets ===')
    
    # Save scaled features and labels as NPZ
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': np.array(feature_cols),
    }
    
    npz_path = os.path.join(features_dir, 'dataset_scaled.npz')
    np.savez_compressed(npz_path, **datasets)
    print(f'[+] Saved scaled dataset to dataset_scaled.npz')
    
    # Save original unscaled data for reference
    datasets_unscaled = {
        'X_train': X[:X_train.shape[0]],
        'X_test': X[X_train.shape[0]:],
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': np.array(feature_cols),
    }
    
    npz_unscaled_path = os.path.join(features_dir, 'dataset_unscaled.npz')
    np.savez_compressed(npz_unscaled_path, **datasets_unscaled)
    print(f'[+] Saved unscaled dataset to dataset_unscaled.npz')
    
    # Save scaler using pickle (more compatible than joblib)
    scaler_path = os.path.join(features_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'[+] Saved fitted scaler to scaler.pkl')
    
    # Save feature names for reference
    feature_names_path = os.path.join(features_dir, 'feature_names.json')
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        json.dump({'feature_names': feature_cols}, f, indent=2)
    print(f'[+] Saved feature names to feature_names.json')
    
    # Save dataset metadata
    metadata_out = {
        'n_samples': len(data),
        'n_features': len(feature_cols),
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'feature_names': feature_cols,
        'feature_means': scaler.mean_.tolist(),
        'feature_stds': scaler.scale_.tolist(),
        'random_state': 42,
        'test_size': 0.2,
        'class_distribution': {
            'benign': int(sum(y == 1)),
            'malicious': int(sum(y == 0))
        }
    }
    
    metadata_path = os.path.join(features_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_out, f, indent=2)
    print(f'[+] Saved dataset metadata to dataset_metadata.json')
    
    # Print summary
    print('\n=== Dataset Summary ===')
    print(f'Total samples: {len(data)}')
    print(f'Training samples: {X_train.shape[0]}')
    print(f'Testing samples: {X_test.shape[0]}')
    print(f'Features: {len(feature_cols)}')
    print(f'Benign samples: {sum(y == 1)}')
    print(f'Malicious samples: {sum(y == 0)}')
    print(f'\nReproducibility: random_state=42')
    print(f'Standardization: StandardScaler (mean=0, std=1)')
    print(f'Scaler saved: {os.path.basename(scaler_path)}')
    
    print('\n[+] Dataset ready for scikit-learn training pipelines!')
    print('    Use X_train, X_test, y_train, y_test from dataset_scaled.npz')
    print('    Load scaler for predictions: pickle.load(open("scaler.pkl", "rb"))')
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler


if __name__ == '__main__':
    main()
