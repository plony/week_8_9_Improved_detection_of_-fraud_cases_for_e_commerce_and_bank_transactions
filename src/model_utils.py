# src/model_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_for_modeling(X, y, numerical_features, categorical_features, test_size=0.25, random_state=42):
    """
    Performs data preprocessing steps for modeling:
    - One-Hot Encoding for categorical features.
    - Train-Test Split (stratified).
    - Scaling numerical features.
    - Handles class imbalance using SMOTE on training data.

    Returns X_train_resampled, X_test, y_train_resampled, y_test
    """
    print("\n--- Starting Preprocessing for Modeling ---")

    # One-Hot Encoding for Categorical Features
    print("Applying One-Hot Encoding...")
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    print(f"One-Hot Encoding complete. New shape: {X.shape}")

    # Align columns after one-hot encoding if X_test is processed separately
    # This is handled naturally if X_test is split from the X after get_dummies

    # Train-Test Split (before scaling and SMOTE)
    print(f"Performing Train-Test Split (test_size={test_size}, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train set class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set class distribution:\n{y_test.value_counts(normalize=True)}")

    # Normalization and Scaling for Numerical Features
    print("Applying StandardScaler to numerical features...")
    scaler = StandardScaler()

    # Identify numerical columns actually present after one-hot encoding
    # Some original numerical features might be dropped or not applicable.
    # We only scale columns that are numerical and exist in the current X_train
    actual_numerical_features = [col for col in numerical_features if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col])]

    if actual_numerical_features:
        X_train[actual_numerical_features] = scaler.fit_transform(X_train[actual_numerical_features])
        X_test[actual_numerical_features] = scaler.transform(X_test[actual_numerical_features])
        print("Scaling complete for numerical features.")
    else:
        print("No numerical features found for scaling.")


    # Handle Class Imbalance (SMOTE on training data only)
    print("Handling Class Imbalance using SMOTE on training data...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Original training data shape: {X_train.shape}")
    print(f"Resampled training data shape: {X_train_resampled.shape}")
    print(f"Class distribution in resampled training data:\n{y_train_resampled.value_counts(normalize=True)}")
    print("Class imbalance handled for training data.")

    print("--- Preprocessing for Modeling Complete ---")
    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist() # Return final feature names for SHAP


def evaluate_model(model, X_test, y_test, model_name, figsize=(7, 6)):
    """
    Evaluates a classification model using appropriate metrics for imbalanced data.
    Prints F1-score, AUC-PR, Confusion Matrix, and plots Precision-Recall Curve.
    """
    print(f"\n--- Evaluating {model_name} ---")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-Score: {f1:.4f}")

    # Precision-Recall Curve and AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    average_precision = average_precision_score(y_test, y_proba) # Alternative way to get AUC-PR

    print(f"AUC-PR (Area Under Precision-Recall Curve): {auc_pr:.4f}")
    print(f"Average Precision Score: {average_precision:.4f}") # Should be similar to auc_pr

    # Optional: ROC-AUC (less reliable for extreme imbalance but good to know)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

    return {'f1_score': f1, 'auc_pr': auc_pr, 'roc_auc': roc_auc, 'confusion_matrix': cm.tolist()}


if __name__ == '__main__':
    # Example usage for testing model_utils.py
    print("--- Testing model_utils.py ---")

    # Create dummy data for testing
    data_size = 1000
    np.random.seed(42)
    X_dummy = pd.DataFrame({
        'num_feature_1': np.random.rand(data_size) * 100,
        'num_feature_2': np.random.randn(data_size) * 10,
        'cat_feature_A': np.random.choice(['X', 'Y', 'Z'], data_size),
        'cat_feature_B': np.random.choice(['P', 'Q'], data_size),
    })
    # Create an imbalanced target variable
    y_dummy = pd.Series(np.zeros(data_size, dtype=int))
    num_fraud = int(data_size * 0.05) # 5% fraud
    fraud_indices = np.random.choice(data_size, num_fraud, replace=False)
    y_dummy.iloc[fraud_indices] = 1

    numerical_cols = ['num_feature_1', 'num_feature_2']
    categorical_cols = ['cat_feature_A', 'cat_feature_B']

    print("\nOriginal Dummy Data (X_dummy head):")
    print(X_dummy.head())
    print("\nOriginal Dummy Data (y_dummy distribution):")
    print(y_dummy.value_counts())

    X_train_resampled, X_test, y_train_resampled, y_test, feature_names = preprocess_for_modeling(
        X_dummy.copy(), y_dummy.copy(), numerical_cols, categorical_cols
    )

    print("\n--- Testing Model Evaluation ---")
    from sklearn.linear_model import LogisticRegression
    dummy_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
    dummy_model.fit(X_train_resampled, y_train_resampled)
    eval_results = evaluate_model(dummy_model, X_test, y_test, "Dummy Logistic Regression")
    print("\nEvaluation Results:", eval_results)