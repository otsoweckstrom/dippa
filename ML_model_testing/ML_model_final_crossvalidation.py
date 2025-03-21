
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the dataset
final_dataset = pd.read_csv("Synthetic_CAD_Data.csv")  # Replace with your dataset file if needed

# Drop unnecessary features (bounding box-related ones)
X = final_dataset.drop(columns=["file_name", "profitable",
                                "bounding_box_x", "bounding_box_y",
                                "bounding_box_z", "bounding_box_volume"])

y = final_dataset["profitable"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set (for balancing)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,        # Fewer trees to prevent overfitting
        max_depth=5,            # Limit depth to avoid memorization
        min_samples_split=5,    # Minimum samples to split a node
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# Store results
results = {}
feature_importances = {}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)  # Train with resampled data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for ROC-AUC

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    }
    
    # Store feature importance for Random Forest and XGBoost
    if name in ["Random Forest", "XGBoost"]:
        feature_importances[name] = model.feature_importances_

# Perform cross-validation (5-fold) for Random Forest and XGBoost
cv_results = {}
for name, model in models.items():
    if name in ["Random Forest", "XGBoost"]:  # Only cross-validate tree-based models
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
        cv_results[name] = {
            "Mean F1-score": np.mean(cv_scores),
            "Std Dev": np.std(cv_scores)
        }

# Convert results to DataFrame for readability
results_df = pd.DataFrame(results).T
cv_results_df = pd.DataFrame(cv_results).T

# Print results
print("Model Performance Metrics:")
print(results_df)
print("Cross-Validation Results:")
print(cv_results_df)

# Plot feature importance for Random Forest and XGBoost
for model_name in ["Random Forest", "XGBoost"]:
    if model_name in feature_importances:
        importance_values = feature_importances[model_name]
        feature_names = X.columns
        sorted_idx = np.argsort(importance_values)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importance_values[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title(f"{model_name} Feature Importance")
        plt.show()
