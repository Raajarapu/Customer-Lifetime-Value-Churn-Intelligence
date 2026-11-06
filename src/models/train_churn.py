# ============================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 6: CUSTOMER CHURN PREDICTION USING LIGHTGBM
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed"

MODEL_PATH = PROJECT_ROOT / "models"

REPORT_PATH = PROJECT_ROOT / "reports/evaluation"

FIGURE_PATH = PROJECT_ROOT / "reports/figures"

for path in [MODEL_PATH, REPORT_PATH, FIGURE_PATH]:
    path.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("STAGE 6: LIGHTGBM CHURN PREDICTION")
print("=" * 65)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv(DATA_PATH / "X_train.csv")

X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()

y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

y_train = y_train.astype(int)

y_test = y_test.astype(int)


print("\nDataset Loaded Successfully")

print("Training Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)

print("Training Churn Distribution:")

print(y_train.value_counts())


# ============================================================
# 3. BASELINE MODEL
# ============================================================

print("\n" + "=" * 65)

print("BASELINE MODEL EVALUATION")

print("=" * 65)


baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_results = {
    "Accuracy": accuracy_score(
        y_test, baseline_predictions
    ),

    "Precision": precision_score(
        y_test, baseline_predictions,
        zero_division=0
    ),

    "Recall": recall_score(
        y_test, baseline_predictions,
        zero_division=0
    ),

    "F1 Score": f1_score(
        y_test, baseline_predictions,
        zero_division=0
    )
}

print("\nBaseline Results:")

for metric, value in baseline_results.items():

    print(f"{metric}: {value:.4f}")


# ============================================================
# 4. LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 65)

print("LIGHTGBM MODEL TRAINING")

print("=" * 65)


model = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=300,

    learning_rate=0.05,

    max_depth=7,

    num_leaves=31,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    class_weight="balanced",

    random_state=42,

    verbosity=-1,

    n_jobs=-1
)


model.fit(X_train, y_train)

print("\nLightGBM Training Completed!")


# ============================================================
# 5. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 65)

print("5-FOLD CROSS VALIDATION")

print("=" * 65)


cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


cv_results = cross_validate(

    model,

    X_train,

    y_train,

    cv=cv,

    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision"
    ],

    n_jobs=-1

)


cv_report = {}

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision"
]:

    scores = cv_results[f"test_{metric}"]

    cv_report[metric] = {
        "Mean": scores.mean(),
        "Std": scores.std()
    }

    print(
        f"{metric}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )


# ============================================================
# 6. TEST DATA PREDICTIONS
# ============================================================

print("\n" + "=" * 65)

print("TEST DATA EVALUATION")

print("=" * 65)


predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

results = {

    "Accuracy": accuracy_score(
        y_test, predictions
    ),

    "Precision": precision_score(
        y_test, predictions,
        zero_division=0
    ),

    "Recall": recall_score(
        y_test, predictions,
        zero_division=0
    ),

    "F1 Score": f1_score(
        y_test, predictions,
        zero_division=0
    ),

    "ROC-AUC": roc_auc_score(
        y_test, probabilities
    ),

    "PR-AUC": average_precision_score(
        y_test, probabilities
    )

}


print("\nLIGHTGBM PERFORMANCE:")

for metric, value in results.items():

    print(f"{metric}: {value:.4f}")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Retained",
            "Churned"
        ],
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Retained",
        "Churned"
    ]

)

disp.plot()

plt.title("LightGBM Confusion Matrix")

plt.savefig(

    FIGURE_PATH / "10_lightgbm_confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 10. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(

    y_test,

    probabilities

)

plt.title("LightGBM ROC Curve")

plt.savefig(

    FIGURE_PATH / "11_lightgbm_roc_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 11. PRECISION RECALL CURVE
# ============================================================

PrecisionRecallDisplay.from_predictions(

    y_test,

    probabilities

)

plt.title("LightGBM Precision-Recall Curve")

plt.savefig(

    FIGURE_PATH / "12_lightgbm_pr_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTOP 15 IMPORTANT FEATURES")

print(feature_importance.head(15))


feature_importance.to_csv(

    REPORT_PATH / "feature_importance.csv",

    index=False

)


plt.figure(figsize=(10, 7))

top_features = feature_importance.head(15).sort_values(
    "Importance"
)

plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)

plt.title("Top 15 LightGBM Feature Importances")

plt.xlabel("Importance")

plt.savefig(

    FIGURE_PATH / "13_feature_importance.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(

    model,

    MODEL_PATH / "lightgbm_churn_model.pkl"

)


# ============================================================
# 14. SAVE EVALUATION RESULTS
# ============================================================

evaluation_report = pd.DataFrame(

    [baseline_results, results],

    index=[
        "Baseline",
        "LightGBM"
    ]

)

evaluation_report.to_csv(

    REPORT_PATH / "model_evaluation.csv"

)


# Save cross-validation results

pd.DataFrame(cv_report).T.to_csv(

    REPORT_PATH / "cross_validation_results.csv"

)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("STAGE 6 COMPLETED")

print("=" * 65)

print("\nBaseline vs LightGBM:")

print(evaluation_report)

print("\nModel Saved:")

print(MODEL_PATH / "lightgbm_churn_model.pkl")

print("\nEvaluation Reports Saved Successfully!")

print("\nGenerated Visualizations:")

print("1. Confusion Matrix")

print("2. ROC Curve")

print("3. Precision-Recall Curve")

print("4. Feature Importance")

print("\n" + "=" * 65)