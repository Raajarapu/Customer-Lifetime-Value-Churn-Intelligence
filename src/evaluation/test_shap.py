import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

X = pd.read_csv(ROOT / "data/processed/X_test.csv")

model = joblib.load(
    ROOT / "models/lightgbm_churn_model.pkl"
)

X_sample = X.head(200)

explainer = shap.TreeExplainer(model)

values = explainer.shap_values(X_sample)

if isinstance(values, list):
    values = values[1]

output = ROOT / "reports/figures"

output.mkdir(parents=True, exist_ok=True)

features = [
    "numerical__Age",
    "numerical__Balance",
    "numerical__CreditScore",
    "numerical__Tenure"
]

for feature in features:

    print(f"Generating SHAP plot: {feature}")

    plt.figure(figsize=(10, 6))

    shap.dependence_plot(
        feature,
        values,
        X_sample,
        interaction_index=None,
        show=False
    )

    plt.title(f"SHAP Dependence Plot: {feature}")

    filename = feature.replace("numerical__", "")

    plt.savefig(
        output / f"shap_dependence_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Successfully saved: {filename}")

print("\nAll SHAP dependence plots generated!")