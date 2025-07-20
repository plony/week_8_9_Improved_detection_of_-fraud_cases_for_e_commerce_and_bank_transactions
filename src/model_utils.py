# src/model_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_for_modeling(X, y, numerical_features, categorical_features, test_size=0.2, random_state=42):
    """
    Preprocesses the data for modeling:
    1. Splits data into training and testing sets.
    2. Applies One-Hot Encoding to categorical features.
    3. Scales numerical features using StandardScaler.
    4. Applies SMOTE to the training data for imbalanced classes.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        numerical_features (list): List of column names that are numerical.
        categorical_features (list): List of column names that are categorical.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, feature_names)
    """
    print("\n--- Starting Preprocessing for Modeling ---")

    # Train-Test Split (before scaling and SMOTE)
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Create a preprocessor pipeline for numerical and categorical features
    # Numerical features are scaled
    # Categorical features are one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any, though none expected here)
    )

    print("Applying preprocessing (scaling and one-hot encoding)...")
    # Fit and transform on training data, transform only on test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    # This is a bit complex with ColumnTransformer, but crucial for SHAP/LGBM with named features
    # Get feature names from one-hot encoder
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    # Combine numerical and one-hot encoded feature names
    all_feature_names = numerical_features + list(ohe_feature_names)

    print(f"Preprocessing complete. X_train_processed shape: {X_train_processed.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}")

    # Apply SMOTE only to the training data
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"SMOTE complete. X_train_resampled shape: {X_train_resampled.shape}, y_train_resampled shape: {y_train_resampled.shape}")
    print(f"Original training class distribution:\n{y_train.value_counts()}")
    print(f"Resampled training class distribution:\n{y_train_resampled.value_counts()}")

    print("--- Preprocessing for Modeling Complete ---")
    return X_train_resampled, X_test_processed, y_train_resampled, y_test, all_feature_names


def train_model(model_type, X_train, y_train, **kwargs):
    """
    Trains a specified machine learning model.

    Args:
        model_type (str): Type of model to train ('logistic_regression', 'lightgbm').
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        **kwargs: Additional parameters for the model.

    Returns:
        sklearn.base.BaseEstimator: The trained model.
    """
    print(f"Training {model_type} model...")
    if model_type == 'logistic_regression':
        # Adjust class_weight for imbalanced data, even with SMOTE it can help
        # C is inverse of regularization strength; smaller values specify stronger regularization.
        model = LogisticRegression(random_state=42, solver='liblinear',
                                   class_weight='balanced', max_iter=1000, **kwargs)
    elif model_type == 'lightgbm':
        # Objective 'binary' for binary classification
        # is_unbalance=True or scale_pos_weight for imbalanced data
        # num_leaves, learning_rate, n_estimators are common params
        model = lgb.LGBMClassifier(objective='binary', random_state=42,
                                   n_estimators=1000, learning_rate=0.05,
                                   num_leaves=31,  # Default, can tune
                                   reg_alpha=0.1, reg_lambda=0.1, # L1/L2 regularization
                                   colsample_bytree=0.8, subsample=0.8, # Feature/data subsampling
                                   # Consider using scale_pos_weight if SMOTE isn't enough or for more control
                                   # scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                                   **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    print(f"{model_type} training complete.")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the performance of a trained model and returns key metrics.

    Args:
        model (sklearn.base.BaseEstimator): The trained model.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target.
        model_name (str, optional): Name of the model for print statements. Defaults to "Model".

    Returns:
        tuple: (metrics_dict, y_pred, y_proba)
            metrics_dict (dict): Dictionary of performance metrics.
            y_pred (array): Predicted classes.
            y_proba (array): Predicted probabilities for the positive class.
    """
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    # AUC-PR is crucial for imbalanced datasets
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall_curve, precision_curve)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr,
        'confusion_matrix': confusion_matrix(y_test, y_pred) # Storing for later display
    }

    print(f"{model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    return metrics, y_pred, y_proba

def plot_confusion_matrix(y_true, y_pred, title, ax=None):
    """
    Plots the confusion matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues', ax=ax)
    ax.set_title(title)
    return ax

def plot_roc_pr_curves(y_true, y_proba, model_name, figsize=(12, 5)):
    """
    Plots ROC AUC and Precision-Recall curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    roc_display = RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    axes[0].set_title(f'ROC Curve for {model_name}')
    axes[0].legend()
    axes[0].grid(True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, label=f'{model_name} (AUC-PR = {pr_auc:.2f})', color='orange')
    # Baseline for PR curve (proportion of positive class)
    no_skill = len(y_true[y_true==1]) / len(y_true)
    axes[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='gray')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve for {model_name}')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Testing model_utils.py")

    # Create dummy data for testing preprocessing and model functions
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=0,
        n_clusters_per_class=1, weights=[0.95], flip_y=0, random_state=42
    )
    # Convert to DataFrame to mimic real data with named columns
    X_dummy = pd.DataFrame(X_dummy, columns=[f'num_feat_{i}' for i in range(10)])
    # Add some categorical features
    X_dummy['cat_feat_A'] = np.random.choice(['A', 'B', 'C'], size=1000)
    X_dummy['cat_feat_B'] = np.random.choice(['X', 'Y'], size=1000)

    numerical_cols = [f'num_feat_{i}' for i in range(10)]
    categorical_cols = ['cat_feat_A', 'cat_feat_B']

    # Test preprocess_for_modeling
    print("\n--- Testing Preprocessing ---")
    X_train_res, X_test_proc, y_train_res, y_test_orig, feature_names = \
        preprocess_for_modeling(X_dummy.copy(), pd.Series(y_dummy).copy(),
                                numerical_cols, categorical_cols)
    print(f"Feature names after preprocessing: {feature_names[:12]}...") # Show first few

    # Test train_model and evaluate_model
    print("\n--- Testing Model Training and Evaluation ---")
    # Logistic Regression
    lr_model = train_model('logistic_regression', X_train_res, y_train_res)
    metrics_lr, y_pred_lr, y_proba_lr = evaluate_model(lr_model, X_test_proc, y_test_orig, 'Dummy LR')
    plot_confusion_matrix(y_test_orig, y_pred_lr, 'Dummy LR Confusion Matrix')
    plot_roc_pr_curves(y_test_orig, y_proba_lr, 'Dummy LR')

    # LightGBM
    lgbm_model = train_model('lightgbm', X_train_res, y_train_res)
    metrics_lgbm, y_pred_lgbm, y_proba_lgbm = evaluate_model(lgbm_model, X_test_proc, y_test_orig, 'Dummy LightGBM')
    plot_confusion_matrix(y_test_orig, y_pred_lgbm, 'Dummy LightGBM Confusion Matrix')
    plot_roc_pr_curves(y_test_orig, y_proba_lgbm, 'Dummy LightGBM')
    plt.show()