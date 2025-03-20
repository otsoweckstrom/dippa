import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the dataset
final_dataset = pd.read_csv("Synthetic_CAD_Data.csv")  # Replace with your dataset file if needed

# Drop non-numeric columns
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

print("Random Forest Feature Importance:")
print(pd.DataFrame({"Feature": feature_names, "Importance": importance_values}))


from sklearn.metrics import classification_report

# Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,   # Increase trees for stability
    max_depth=5,        # Reduce depth to control overfitting
    min_samples_split=10,  # Reduce overfitting by requiring more samples per split
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predictions on training and test sets
y_train_pred = rf_model.predict(X_train_resampled)
y_test_pred = rf_model.predict(X_test)

y_train_prob = rf_model.predict_proba(X_train_resampled)[:, 1]
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate performance
train_metrics = {
    "Accuracy": accuracy_score(y_train_resampled, y_train_pred),
    "Precision": precision_score(y_train_resampled, y_train_pred),
    "Recall": recall_score(y_train_resampled, y_train_pred),
    "F1-score": f1_score(y_train_resampled, y_train_pred),
    "AUC-ROC": roc_auc_score(y_train_resampled, y_train_prob)
}

test_metrics = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1-score": f1_score(y_test, y_test_pred),
    "AUC-ROC": roc_auc_score(y_test, y_test_prob)
}

# Display results
print("**Train Performance Metrics:**")
print(pd.DataFrame(train_metrics, index=["Train"]).T)

print("\n**Test Performance Metrics:**")
print(pd.DataFrame(test_metrics, index=["Test"]).T)

# Classification reports for deeper analysis
print("\nClassification Report (Train):\n", classification_report(y_train_resampled, y_train_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))


import numpy as np

# Predict probabilities instead of labels
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Adjust classification threshold
final_threshold = 0.6  # Set to 0.7 if you prioritize high precision
y_final_pred = (y_test_prob >= final_threshold).astype(int)

print("\nFinal Model Evaluation:")
print("Precision:", precision_score(y_test, y_final_pred))
print("Recall:", recall_score(y_test, y_final_pred))
print("F1-score:", f1_score(y_test, y_final_pred))
