#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURATION (edit here)
# =========================

DATA_DIR = "classical_ml_dataset"  # Directory with features_train.csv, etc.
RANDOM_SEED = 42  # For reproducible splitting if no val set
TEST_SIZE = 0.1  # Split fraction if no val set
N_JOBS = -1  # Parallel jobs for GridSearchCV (use all cores)

# RF Hyperparameter grid (focused on key params for F1 optimization)
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Scoring for imbalance: weighted F1 (accounts for support per class)
SCORING = 'f1_weighted'

# =========================
# HELPER FUNCTIONS
# =========================

def load_data(split_name: str) -> pd.DataFrame:
    """
    Load features CSV for a split.
    Drops metadata if present, ensures 'label' is float/int.
    """
    filepath = os.path.join(DATA_DIR, f'features_{split_name}.csv')
    if not os.path.exists(filepath):
        print(f"⚠ No {split_name} file found: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    
    # Drop metadata columns if they exist
    meta_cols = ['frame_num', 'source_file']
    df = df.drop(columns=[col for col in meta_cols if col in df.columns])
    
    # Ensure label is numeric
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    
    # Features: all except 'label'
    feature_cols = [col for col in df.columns if col != 'label']
    
    print(f"✓ Loaded {split_name}: {len(df)} samples, features: {len(feature_cols)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df[feature_cols + ['label']]

def split_if_no_val(X_train_full, y_train_full, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    If no val set, split train into train/val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=random_state, stratify=y_train_full
    )
    return X_train, X_val, y_train, y_val

def tune_hyperparams(X_train, y_train, X_val, y_val):
    """
    Grid search for best RF params using val set (or CV if no val).
    """
    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS)
    grid_search = GridSearchCV(
        rf, PARAM_GRID, cv=5, scoring=SCORING, n_jobs=N_JOBS, verbose=1
    )
    
    # Fit on train, score on val (GridSearchCV uses CV internally on train)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    
    print(f"✓ Best params: {grid_search.best_params_}")
    print(f"✓ Best CV {SCORING}: {best_score:.4f}")
    
    # Evaluate on val if provided
    if X_val is not None:
        val_pred = best_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        print(f"✓ Val {SCORING}: {val_f1:.4f}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate on test set: F1, report, confusion matrix.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Test {SCORING}: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (People Count)')
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return f1

# =========================
# MAIN FUNCTION
# =========================

def main():
    """
    Main execution: load data, tune, evaluate.
    """
    print("="*80)
    print("RANDOM FOREST FOR RADAR PEOPLE COUNTING (F1-Optimized)")
    print("="*80)
    print(f"Data dir: {DATA_DIR}")
    print(f"Scoring: {SCORING} (for imbalance)")
    print(f"Hyperparam grid size: {np.prod([len(v) for v in PARAM_GRID.values()])} combinations")
    print("="*80)
    
    # Load datasets
    train_df = load_data('train')
    val_df = load_data('val')
    test_df = load_data('test')
    
    if train_df.empty:
        print("❌ No training data found. Run 9-dataset_generator_classicalml.py first.")
        return
    
    if test_df.empty:
        print("⚠ No test data; skipping evaluation.")
    
    # Prepare features and labels
    feature_cols = [col for col in train_df.columns if col != 'label']
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df['label'].values
    
    X_test = test_df[feature_cols].values if not test_df.empty else None
    y_test = test_df['label'].values if not test_df.empty else None
    
    # Handle validation
    if not val_df.empty:
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values
        X_train = X_train_full
        y_train = y_train_full
        print("✓ Using existing val set for tuning.")
    else:
        X_train, X_val, y_train, y_val = split_if_no_val(X_train_full, y_train_full)
        print(f"✓ No val set; split {TEST_SIZE*100}% from train (seed={RANDOM_SEED}).")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Tune hyperparameters
    best_model = tune_hyperparams(X_train, y_train, X_val, y_val)
    
    # Evaluate on test
    if X_test is not None:
        final_f1 = evaluate_model(best_model, X_test, y_test)
        print(f"\n✓ Final model F1 (test): {final_f1:.4f}")
    
    # Save model (optional)
    import joblib
    joblib.dump(best_model, os.path.join(DATA_DIR, 'best_rf_model.pkl'))
    print(f"\n✓ Model saved: best_rf_model.pkl in {DATA_DIR}")
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE! Use best_rf_model.pkl for predictions.")
    print("="*80)

if __name__ == "__main__":
    main()
