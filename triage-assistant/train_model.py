"""
Health Tech AI/ML Project: Breast Cancer Risk Triage Model
------------------------------------------------------------
Goal: Train a classifier that predicts whether a tumor is malignant or
benign based on diagnostic measurements, then export it in a form we
can embed in a lightweight web demo (no server needed).

Dataset: Wisconsin Breast Cancer Diagnostic dataset (bundled with
scikit-learn, so it's a real, widely-used clinical ML benchmark).
"""

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target  # y: 0 = malignant, 1 = benign
feature_names = list(data.feature_names)

# 2. Pick a small, interpretable subset of features for the demo UI
#    (using the full 30 features would make for a terrible interface)
demo_features = [
    "mean radius", "mean texture", "mean perimeter",
    "mean area", "mean concavity", "mean smoothness"
]
demo_idx = [feature_names.index(f) for f in demo_features]

X_demo = X[:, demo_idx]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_demo, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features (logistic regression is sensitive to feature scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train logistic regression
#    class_weight='balanced' + we'll examine recall on malignant class
#    specifically, since in health triage missing a malignant case
#    (false negative) is far more costly than a false alarm.
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
# Malignant = 0, so precision/recall "for malignant" means pos_label=0
precision_malignant = precision_score(y_test, y_pred, pos_label=0)
recall_malignant = recall_score(y_test, y_pred, pos_label=0)
f1_malignant = f1_score(y_test, y_pred, pos_label=0)
cm = confusion_matrix(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision (malignant): {precision_malignant:.3f}")
print(f"Recall (malignant):    {recall_malignant:.3f}  <-- most important for triage")
print(f"F1 (malignant):        {f1_malignant:.3f}")
print("\nConfusion matrix (rows=actual, cols=predicted) [malignant, benign]:")
print(cm)
print("\n", classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

# 7. Export everything needed to reproduce this model's predictions
#    in plain JavaScript (no Python server required for the demo).
export = {
    "feature_names": demo_features,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coefficients": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "classes": ["malignant", "benign"],  # index 0 = malignant per sklearn encoding
    "metrics": {
        "accuracy": acc,
        "precision_malignant": precision_malignant,
        "recall_malignant": recall_malignant,
        "f1_malignant": f1_malignant,
        "confusion_matrix": cm.tolist(),
        "test_set_size": int(len(y_test))
    },
    # feature min/max across the whole dataset, so the demo UI can set
    # sensible slider ranges
    "feature_ranges": {
        f: [float(X[:, feature_names.index(f)].min()), float(X[:, feature_names.index(f)].max())]
        for f in demo_features
    }
}

with open("model_export.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nExported model to model_export.json")
