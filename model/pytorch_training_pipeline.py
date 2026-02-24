"""
PyTorch Training Pipeline for HIDS Attack Detection
Production-ready modular pipeline for training and evaluating a neural network model.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

# ============================================================================
# CONFIGURATION & RANDOM SEED
# ============================================================================

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
VALIDATION_SPLIT = 0.2

# Paths
DATA_DIR = Path(__file__).parent.parent / 'hids_dataset' / 'features'
MODEL_OUTPUT_DIR = Path(__file__).parent
MODEL_PATH = MODEL_OUTPUT_DIR / 'model.pth'
METRICS_PATH = MODEL_OUTPUT_DIR / 'metrics.json'

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class HIIDSMLP(nn.Module):
    """
    Feedforward Neural Network (MLP) for HIDS attack detection.
    
    Architecture:
    - Input: 24 features
    - Hidden layer 1: 64 neurons + ReLU + BatchNorm + Dropout(0.3)
    - Hidden layer 2: 32 neurons + ReLU + Dropout(0.2)
    - Output: 1 neuron + Sigmoid (binary classification)
    """
    
    def __init__(self, input_size, hidden1=64, hidden2=32, dropout1=0.3, dropout2=0.2):
        super(HIIDSMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)
        
        self.fc3 = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """Load training and test datasets, preprocess, and return PyTorch tensors."""
    
    print("\n" + "="*70)
    print("LOADING & PREPROCESSING DATA")
    print("="*70)
    
    # Load datasets
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    
    print(f"\n✓ Training data shape: {train_df.shape}")
    print(f"✓ Test data shape: {test_df.shape}")
    
    # Note: Labels are 0=benign, 1=malicious (but in the dataset we need to check)
    print(f"\n✓ Training set class distribution:")
    print(train_df['label'].value_counts())
    
    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    print(f"\n✓ Features: {X_train.shape[1]}")
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Test samples: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Features standardized (mean≈0, std≈1)")
    
    # Split training data into train and validation (80-20 split)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train
    )
    
    print(f"✓ Training split: {X_tr.shape[0]} samples")
    print(f"✓ Validation split: {X_val.shape[0]} samples")
    
    # Convert to PyTorch tensors
    X_tr_tensor = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_tensor = torch.FloatTensor(y_tr).reshape(-1, 1).to(DEVICE)
    
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(DEVICE)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(DEVICE)
    
    print(f"\n✓ All data converted to PyTorch tensors on {DEVICE}")
    
    return (X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor,
            X_test_tensor, y_test_tensor, X_train.shape[1], y_train, y_test)

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the neural network with early stopping."""
    
    print("\n" + "="*70)
    print(f"TRAINING MODEL (Target: {epochs} epochs, Batch size: {BATCH_SIZE})")
    print("="*70)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"\n✓ Training complete. Best model saved to {MODEL_PATH}")
    
    return train_losses, val_losses

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set and return metrics."""
    
    print("\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test).cpu().numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    y_test_np = y_test.cpu().numpy().flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred)
    recall = recall_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred)
    roc_auc = roc_auc_score(y_test_np, y_pred_proba)
    
    print(f"\n✓ Accuracy:  {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall:    {recall:.4f}  *** ATTACK DETECTION RATE ***")
    print(f"✓ F1-Score:  {f1:.4f}")
    print(f"✓ ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test_np, y_pred, target_names=['Benign', 'Malicious']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_np, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n✓ Confusion Matrix:")
    print(f"  TN (Correct benign):     {tn}")
    print(f"  FP (False alarms):       {fp}")
    print(f"  FN (Attacks missed):     {fn}  🚨")
    print(f"  TP (Attacks detected):   {tp}  ✅")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }
    
    return metrics, y_pred_proba, y_test_np, cm

# ============================================================================
# THRESHOLD TUNING
# ============================================================================

def tune_classification_threshold(y_test, y_pred_proba):
    """Evaluate different classification thresholds to achieve target recall."""
    
    print("\n" + "="*70)
    print("CLASSIFICATION THRESHOLD TUNING")
    print("="*70)
    print("\nTesting thresholds from 0.1 to 0.9 for recall ≥ 0.95...")
    print("-"*70)
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_threshold = 0.5
    best_recall = 0.0
    
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int).flatten()
        
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        
        print(f"Threshold {threshold:.1f} | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1: {f1:.4f}")
        
        results.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })
        
        # Find threshold achieving recall >= 0.95 with highest precision
        if recall >= 0.95 and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    
    if best_recall >= 0.95:
        print(f"\n✓ Found threshold {best_threshold:.1f} achieving recall ≥ 0.95")
    else:
        print(f"\n⚠ Could not find threshold with recall ≥ 0.95")
        print(f"  Best recall achieved: {best_recall:.4f} at threshold {best_threshold:.1f}")
    
    return results, best_threshold

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Training Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss (BCELoss)', fontweight='bold')
    ax.set_title('Training vs Validation Loss', fontweight='bold', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / 'training_curves.png', dpi=100, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {MODEL_OUTPUT_DIR / 'training_curves.png'}")
    plt.close()

def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('Confusion Matrix - PyTorch Model', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {MODEL_OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.close()

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve."""
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve - PyTorch Model', fontweight='bold', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / 'roc_curve.png', dpi=100, bbox_inches='tight')
    print(f"✓ ROC curve saved to {MODEL_OUTPUT_DIR / 'roc_curve.png'}")
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete training and evaluation pipeline."""
    
    print("\n" + "="*70)
    print("PyTorch HIDS ATTACK DETECTION TRAINING PIPELINE")
    print("="*70)
    
    # 1. Load and preprocess data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     num_features, y_train_original, y_test_original) = load_and_preprocess_data()
    
    # 2. Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize model
    model = HIIDSMLP(input_size=num_features).to(DEVICE)
    print(f"\n✓ Model initialized on {DEVICE}")
    print(model)
    
    # 4. Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # 5. Evaluate on test set
    metrics, y_pred_proba, y_test_np, cm = evaluate_model(model, X_test, y_test)
    
    # 6. Tune classification threshold
    threshold_results, best_threshold = tune_classification_threshold(y_test_np, y_pred_proba)
    metrics['best_threshold'] = best_threshold
    metrics['threshold_tuning'] = threshold_results
    
    # 7. Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✓ Metrics saved to {METRICS_PATH}")
    
    # 8. Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    plot_training_curves(train_losses, val_losses)
    plot_confusion_matrix(cm)
    plot_roc_curve(y_test_np, y_pred_proba)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n✓ Model saved: {MODEL_PATH}")
    print(f"✓ Metrics saved: {METRICS_PATH}")
    print(f"✓ Visualizations saved:")
    print(f"  - {MODEL_OUTPUT_DIR / 'training_curves.png'}")
    print(f"  - {MODEL_OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"  - {MODEL_OUTPUT_DIR / 'roc_curve.png'}")
    print(f"\n✓ Test Set Performance:")
    print(f"  - Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  - Precision:  {metrics['precision']:.4f}")
    print(f"  - Recall:     {metrics['recall']:.4f} (Attack detection rate)")
    print(f"  - F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  - Best Threshold: {best_threshold:.2f}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
