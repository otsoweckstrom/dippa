import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
final_dataset = pd.read_csv("SyntheticFeatures.csv")  # Replace with your dataset file if needed

# Drop non-numeric columns
X = final_dataset.drop(columns=["file_name", "profitable"])
y = final_dataset["profitable"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(
    n_estimators=50,        # Fewer trees to prevent overfitting
    max_depth=5,            # Limit depth to avoid memorization
    min_samples_split=5,     # Minimum samples to split a node
    random_state=42
),
}

# Store results
results = {}
feature_importances = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for ROC-AUC

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    }
    
    # Store feature importance for Random Forest
    if name == "Random Forest":
        feature_importances[name] = model.feature_importances_

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("Model Performance Metrics:")
print(results_df)

# Plot feature importance for Random Forest
if "Random Forest" in feature_importances:
    importance_values = feature_importances["Random Forest"]
    feature_names = X.columns
    sorted_idx = np.argsort(importance_values)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance_values[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()
