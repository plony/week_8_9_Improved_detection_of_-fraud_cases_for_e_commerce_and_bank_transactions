.
├── data/                    # Stores raw and processed datasets, including test sets for models.
│   ├── raw_ecommerce_data.csv
│   ├── raw_bank_data.csv
│   ├── processed_ecommerce_data.pkl # Processed X_test, y_test, feature_names for E-commerce
│   └── processed_bank_data.pkl      # Processed X_test, y_test, feature_names for Bank
├── models/                  # Stores trained machine learning models.
│   ├── ecommerce_best_model.pkl   # Best performing model for E-commerce
│   └── bank_best_model.pkl        # Best performing model for Bank Transactions
├── notebooks/               # Jupyter notebooks for data processing, modeling, and explanation.
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_model_explainability.ipynb
├── .gitignore               # Specifies intentionally untracked files to ignore.
├── requirements.txt         # Lists all Python dependencies.
└── README.md                # This file.