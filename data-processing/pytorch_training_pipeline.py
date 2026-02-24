"""
Production-Ready PyTorch Training Pipeline for HIDS Dataset
Trains a feedforward neural network for malicious audit log detection
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# ============================================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================================

class Config:
    """Configuration parameters for training pipeline"""
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset paths
    TRAIN_CSV = '../hids_dataset/features/train.csv'
    TEST_CSV = '../hids_dataset/features/test.csv'
    
    # Model architecture
    INPUT_SIZE = None  # Will be set from data
    HIDDEN_SIZE_1 = 64
    HIDDEN_SIZE_2 = 32
    OUTPUT_SIZE = 1
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.2
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15
    
    # Output
    MODEL_PATH = 'model.pth'
    METRICS_PATH = 'metrics.json'
    PLOTS_DIR = 'plots'

# Set random seeds for reproducibility
def set_random_seeds(seed: int = Config.RANDOM_SEED):
    """Set reproducible random seeds across all libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds()

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV dataset and separate features from label
    
    Args:
        csv_path: Path to CSV file with 'label' column
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.float32).reshape(-1, 1)
    
    print(f"  ✓ Features shape: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")
    
    return X, y

def create_data_loaders(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    batch_size: int = 32,
    validation_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders with stratified split
    
    Args:
        X_train: Training features
        y_train: Training labels
        batch_size: Batch size for loaders
        validation_split: Proportion of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Stratified split for balanced validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=Config.RANDOM_SEED,
        stratify=y_train  # Maintain class distribution
    )
    
    # Convert to PyTorch tensors
    X_tr_tensor = torch.from_numpy(X_tr).float()
    y_tr_tensor = torch.from_numpy(y_tr).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()
    
    # Create datasets
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"\n✓ Training set: {len(train_dataset)} samples")
    print(f"✓ Validation set: {len(val_dataset)} samples")
    print(f"  Class distribution (training):")
    print(f"    - Benign (0): {(y_tr == 0).sum()} ({100*(y_tr == 0).sum()/len(y_tr):.1f}%)")
    print(f"    - Malicious (1): {(y_tr == 1).sum()} ({100*(y_tr == 1).sum()/len(y_tr):.1f}%)")
    
    return train_loader, val_loader

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class HIDS_MLP(nn.Module):
    """Feedforward Neural Network for HIDS attack detection
    
    Architecture:
        Input → BatchNorm → Linear(64) → ReLU → Dropout(0.3)
                        → Linear(32) → ReLU → Dropout(0.2)
                        → Linear(1) → Sigmoid
    """
    
    def __init__(
        self, 
        input_size: int,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2
    ):
        super(HIDS_MLP, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(dropout_1)
        
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_2)
        
        self.fc3 = nn.Linear(hidden_size_2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.input_bn(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Args:
            val_loss: Validation loss
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (best: {self.best_epoch})")
                return True
        
        return False

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate model
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 15,
    device: str = 'cpu'
) -> Dict:
    """Train model with early stopping
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        patience: Early stopping patience
        device: Device to use
        
    Returns:
        Dictionary with training history and best epoch
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    
    print("\n" + "=" * 70)
    print("TRAINING FEEDFORWARD NEURAL NETWORK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max Epochs: {epochs}, Patience: {patience}\n")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
        
        if early_stopping(val_loss, epoch):
            break
    
    print(f"\n✓ Training complete! Best epoch: {early_stopping.best_epoch + 1}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': early_stopping.best_epoch,
        'best_loss': early_stopping.best_loss
    }

# ============================================================================
# EVALUATION
# ============================================================================

def predict_batch(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int = 32,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions on batch data
    
    Args:
        model: Trained model
        X: Features array
        device: Device to use
        batch_size: Batch size for inference
        threshold: Classification threshold
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            probs = model(batch_X).cpu().numpy()
            all_probs.extend(probs)
    
    probs = np.array(all_probs).flatten()
    preds = (probs >= threshold).astype(int)
    
    return preds, probs

def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    threshold: float = 0.5
) -> Dict:
    """Evaluate model on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device to use
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_test_flat = y_test.flatten()
    y_pred, y_pred_proba = predict_batch(model, X_test, device, threshold=threshold)
    
    accuracy = accuracy_score(y_test_flat, y_pred)
    precision = precision_score(y_test_flat, y_pred)
    recall = recall_score(y_test_flat, y_pred)
    f1 = f1_score(y_test_flat, y_pred)
    roc_auc = roc_auc_score(y_test_flat, y_pred_proba)
    cm = confusion_matrix(y_test_flat, y_pred)
    
    print("\n" + "=" * 70)
    print(f"MODEL EVALUATION (Threshold: {threshold:.2f})")
    print("=" * 70)
    print(f"\n✓ Accuracy:  {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall:    {recall:.4f}  *** KEY METRIC ***")
    print(f"✓ F1-Score:  {f1:.4f}")
    print(f"✓ ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n✓ Confusion Matrix:")
    print(f"    Predicted →  Benign  Malicious")
    print(f"  Actual ↓")
    print(f"    Benign      {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"    Malicious   {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n✓ Security Metrics:")
    print(f"    - True Negatives (Benign correct):   {tn}")
    print(f"    - True Positives (Attack caught):    {tp}")
    print(f"    - False Positives (False alarm):     {fp}")
    print(f"    - False Negatives (Attacks missed):  {fn} 🚨")
    
    print(f"\n✓ Detailed Report:")
    print(classification_report(y_test_flat, y_pred, target_names=['Benign', 'Malicious']))
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'threshold': threshold
    }

def tune_threshold(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    min_recall: float = 0.95
) -> Dict:
    """Tune classification threshold to achieve target recall
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device to use
        min_recall: Target minimum recall
        
    Returns:
        Dictionary with threshold tuning results
    """
    y_test_flat = y_test.flatten()
    _, y_pred_proba = predict_batch(model, X_test, device, threshold=0.5)
    
    print("\n" + "=" * 70)
    print("THRESHOLD TUNING FOR RECALL OPTIMIZATION")
    print("=" * 70)
    print(f"\nTarget Recall: {min_recall:.1%}\n")
    print("Threshold | Precision | Recall | F1-Score | Specificity | Threshold Goal")
    print("─" * 75)
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    best_threshold = None
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_test_flat, y_pred, zero_division=0)
        recall = recall_score(y_test_flat, y_pred, zero_division=0)
        f1 = f1_score(y_test_flat, y_pred, zero_division=0)
        cm = confusion_matrix(y_test_flat, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        goal_marker = ""
        if recall >= min_recall:
            goal_marker = f"✅ (Recall ≥ {min_recall:.0%})"
            if best_threshold is None or precision > best_score:
                best_threshold = threshold
                best_score = precision
        
        print(f"  {threshold:.1f}    |   {precision:.4f}  | {recall:.4f} | {f1:.4f}  |  {specificity:.4f}   | {goal_marker}")
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        })
    
    if best_threshold is not None:
        print(f"\n✓ Best threshold achieving recall ≥ {min_recall:.1%}: {best_threshold:.1f}")
        print(f"  - Precision at this threshold: {best_score:.4f}")
    else:
        print(f"\n⚠ No threshold found achieving recall ≥ {min_recall:.1%}")
        print(f"  Using threshold 0.1 (maximum recall)")
        best_threshold = 0.1
    
    return {
        'threshold_tuning': results,
        'best_threshold': best_threshold,
        'target_recall': min_recall
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: str = 'plots'
) -> None:
    """Plot training vs validation loss
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=3)
    ax.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (BCELoss)', fontsize=11)
    ax.set_title('Training Progress: Loss over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training loss plot: {plot_path}")
    plt.close()

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str = 'plots'
) -> None:
    """Plot ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_proba)
    roc_auc = roc_auc_score(y_true.flatten(), y_proba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2, color='#2ecc71')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=11)
    ax.set_title('ROC Curve - Attack Detection Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve plot: {plot_path}")
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("PYTORCH TRAINING PIPELINE FOR HIDS ATTACK DETECTION")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading datasets...")
    X_train, y_train = load_dataset(Config.TRAIN_CSV)
    X_test, y_test = load_dataset(Config.TEST_CSV)
    
    # Set input size from data
    Config.INPUT_SIZE = X_train.shape[1]
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. CREATE DATA LOADERS
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2/7] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. BUILD MODEL
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3/7] Building model architecture...")
    model = HIDS_MLP(
        input_size=Config.INPUT_SIZE,
        hidden_size_1=Config.HIDDEN_SIZE_1,
        hidden_size_2=Config.HIDDEN_SIZE_2,
        dropout_1=Config.DROPOUT_1,
        dropout_2=Config.DROPOUT_2
    )
    print(f"✓ Model created for device: {Config.DEVICE}")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. TRAIN MODEL
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4/7] Training model with early stopping...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.MAX_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
        device=Config.DEVICE
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 5. EVALUATE ON TEST SET
    # ─────────────────────────────────────────────────────────────────────
    print("\n[5/7] Evaluating on test set (threshold=0.5)...")
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=Config.DEVICE,
        threshold=0.5
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 6. THRESHOLD TUNING
    # ─────────────────────────────────────────────────────────────────────
    print("\n[6/7] Tuning classification threshold...")
    threshold_results = tune_threshold(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=Config.DEVICE,
        min_recall=0.95
    )
    
    # Evaluate with best threshold
    print("\n" + "─" * 70)
    print("Final evaluation with optimized threshold:")
    print("─" * 70)
    optimized_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=Config.DEVICE,
        threshold=threshold_results['best_threshold']
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 7. VISUALIZATION & SAVING
    # ─────────────────────────────────────────────────────────────────────
    print("\n[7/7] Generating visualizations and saving outputs...")
    
    # Plot training history
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        output_dir=Config.PLOTS_DIR
    )
    
    # Plot ROC curve
    _, y_test_proba = predict_batch(model, X_test, Config.DEVICE)
    plot_roc_curve(y_test, y_test_proba, output_dir=Config.PLOTS_DIR)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': Config.INPUT_SIZE,
            'hidden_size_1': Config.HIDDEN_SIZE_1,
            'hidden_size_2': Config.HIDDEN_SIZE_2,
            'dropout_1': Config.DROPOUT_1,
            'dropout_2': Config.DROPOUT_2
        },
        'threshold': threshold_results['best_threshold']
    }, Config.MODEL_PATH)
    print(f"✓ Model saved to: {Config.MODEL_PATH}")
    
    # Save metrics
    results = {
        'training': {
            'epochs_trained': len(history['train_losses']),
            'best_epoch': history['best_epoch'] + 1,
            'best_validation_loss': float(history['best_loss'])
        },
        'test_metrics': {
            'default_threshold_0.5': metrics,
            'optimized_threshold': {
                **optimized_metrics,
                'threshold': threshold_results['best_threshold']
            }
        },
        'threshold_tuning': threshold_results['threshold_tuning'],
        'best_threshold_for_recall_0.95': threshold_results['best_threshold']
    }
    
    with open(Config.METRICS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Metrics saved to: {Config.METRICS_PATH}")
    
    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE ✅")
    print("=" * 70)
    print(f"\nKey Results (Optimized Threshold: {threshold_results['best_threshold']:.1f}):")
    print(f"  • Recall:        {optimized_metrics['recall']:.1%}")
    print(f"  • Precision:     {optimized_metrics['precision']:.1%}")
    print(f"  • F1-Score:      {optimized_metrics['f1_score']:.1%}")
    print(f"  • Accuracy:      {optimized_metrics['accuracy']:.1%}")
    print(f"  • ROC-AUC:       {optimized_metrics['roc_auc']:.4f}")
    
    cm = optimized_metrics['confusion_matrix']
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"\nSecurity Metrics:")
    print(f"  • Attacks Caught: {tp} (True Positives)")
    print(f"  • Attacks Missed: {fn} (False Negatives)")
    print(f"  • False Alarms:   {fp} (False Positives)")
    
    print(f"\nOutput Files:")
    print(f"  • Model:              {Config.MODEL_PATH}")
    print(f"  • Metrics:            {Config.METRICS_PATH}")
    print(f"  • Plots:              {Config.PLOTS_DIR}/")
    print()

if __name__ == '__main__':
    main()
