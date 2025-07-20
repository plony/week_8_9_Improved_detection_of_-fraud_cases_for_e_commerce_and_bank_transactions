# src/model_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_for_modeling(X: pd.DataFrame, y: pd.Series, numerical_features: list, categorical_features: list, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocesses data for modeling, including splitting, scaling, one-hot encoding, and SMOTE.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train_resampled, X_test_processed, y_train_resampled, y_test, all_feature_names
    """
    print("--- Starting Preprocessing for Modeling ---")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"Splitting data into training and testing sets...")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Create a preprocessor using ColumnTransformer
    # Conditionally add transformers based on whether feature lists are empty
    transformers = []
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    if categorical_features:
        # handle_unknown='ignore' is crucial for unseen categories in test set
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    if not transformers:
        # Handle case where both numerical_features and categorical_features are empty
        raise ValueError("No features provided for preprocessing (both numerical and categorical lists are empty).")

    preprocessor = ColumnTransformer(transformers, remainder='passthrough') # 'passthrough' keeps other columns if any

    print("Applying preprocessing (scaling and one-hot encoding)...")
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    all_feature_names = []
    if numerical_features:
        all_feature_names.extend(numerical_features)
    if categorical_features:
        # Only call get_feature_names_out if the 'cat' transformer was actually included
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names.extend(ohe_feature_names)

    # Convert processed arrays back to DataFrame for better handling (especially for SMOTE and model training)
    X_train_processed = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)


    print(f"Preprocessing complete. X_train_processed shape: {X_train_processed.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}")

    # Apply SMOTE to training data
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    print("SMOTE complete. X_train_resampled shape: {}, y_train_resampled shape: {}".format(X_train_resampled.shape, y_train_resampled.shape))
    print("Original training class distribution:")
    print(y_train.value_counts())
    print("Resampled training class distribution:")
    print(y_train_resampled.value_counts())
    print("--- Preprocessing for Modeling Complete ---")

    return X_train_resampled, X_test_processed, y_train_resampled, y_test, all_feature_names

def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a classification model.

    Args:
        model_name (str): Name of the model to train ('logistic_regression' or 'lightgbm').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        model: Trained model object.
    """
    print(f"\nTraining {model_name} model...")
    if model_name == 'logistic_regression':
        model = LogisticRegression(random_state=42, solver='liblinear', n_jobs=-1)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31,
                               n_jobs=-1,  # Use all available cores
                               # Use 'binary' for binary classification
                               objective='binary',
                               metric='auc', # AUC is a good metric for imbalanced data
                               is_unbalance=True # Helps LightGBM with imbalanced data
                               )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, y_train)
    print(f"{model_name} model training complete.")
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"):
    """
    Evaluates a trained classification model.

    Args:
        model: Trained model object.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for printing results.

    Returns:
        tuple: Dictionary of metrics, predicted labels, predicted probabilities.
    """
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Calculate AUC-PR (Area Under Precision-Recall Curve)
    # This is often more informative for highly imbalanced datasets than ROC AUC.
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recalls, precisions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr
    }

    print(f"--- {model_name} Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return metrics, y_pred, y_proba

def plot_confusion_matrix(y_true, y_pred, title, ax):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {title}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_aspect('equal', adjustable='box')


def plot_roc_pr_curves(y_test, y_proba, title):
    """Plots ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'ROC and Precision-Recall Curves for {title}', fontsize=16)

    # ROC Curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_proba)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # Precision-Recall Curve
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_proba)
    auc_pr = sklearn.metrics.auc(recall, precision)
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {auc_pr:.2f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Import sklearn for plotting functions in the global scope if not already
import sklearn.metrics