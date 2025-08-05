import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns 
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Load the dataset
final_dataset = pd.read_csv("Synthetic_CAD_Data.csv") 

# --- Data Preparation ---
# Define features (X) and target (y)
X = final_dataset.drop(columns=["file_name", "profitable",
                                "bounding_box_x", "bounding_box_y",
                                "bounding_box_z", "bounding_box_volume"])
y = final_dataset["profitable"]
feature_names = X.columns.tolist() # Store feature names for later use

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# Scale the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE 
print(f"Original training set shape: {X_train_scaled.shape}")
print(f"Original training set distribution:\n{y_train.value_counts()}")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Resampled training set shape: {X_train_resampled.shape}")
print(f"Resampled training set distribution:\n{y_train_resampled.value_counts()}")
# --- End Data Preparation ---

# --- Model Training & Evaluation ---
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced_subsample' 
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

results = {}
feature_importances = {}
confusion_matrices = {}

print("\n--- Training and Evaluating Models ---")
for name, model in models.items():
    print(f"Training {name}...")
    # Train on the RESAMPLED (SMOTE'd) data
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on the original, scaled TEST data
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability estimates for ROC-AUC

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred), # Precision for the positive class (1)
        "Recall": recall_score(y_test, y_pred),       # Recall for the positive class (1)
        "F1-score": f1_score(y_test, y_pred),         # F1 for the positive class (1)
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    }
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

  
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
        print(f"Feature importances captured for {name}.")
    elif hasattr(model, 'coef_'):
   
         pass

# --- Cross-Validation ---
print("\n--- Performing Cross-Validation  ---")
cv_results = {}
for name, model in models.items():
    if name in ["Random Forest", "XGBoost"]: 
        print(f"Cross-validating {name}...")
    
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='f1') 
        cv_results[name] = {
            "Mean F1-score": np.mean(cv_scores),
            "Std Dev": np.std(cv_scores)
        }

# --- Output Results ---
results_df = pd.DataFrame(results).T
cv_results_df = pd.DataFrame(cv_results).T

print("\n--- Model Performance Metrics on Test Set ---")
print(results_df)
print("\n--- Cross-Validation F1-Score Results (on Resampled Train Set) ---")
print(cv_results_df)
print("\n--- Confusion Matrices on Test Set ---")
for name, cm in confusion_matrices.items():
    print(f"\nConfusion Matrix for {name}:")
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Not Profitable', 'Predicted Profitable'], yticklabels=['Actual Not Profitable', 'Actual Profitable'])
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


# --- Plot Feature Importance ---
print("\n--- Plotting Feature Importances ---")
for model_name in ["Random Forest", "XGBoost"]:
    if model_name in feature_importances:
        importance_values = feature_importances[model_name]
       
        sorted_idx = np.argsort(importance_values)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importance_values[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout() 
        plt.show()


corr = X.corr()
plt.figure(figsize=(12, 10))  
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

print(f"Test set distribution:\n{y_test.value_counts()}")

print("\n--- Script Finished ---")




