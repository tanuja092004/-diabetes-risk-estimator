"""
Train the diabetes logistic-regression model exactly as done in
Python_Implementation.ipynb and export everything the web app needs
(coefficients, scaler ranges, medians, metrics, chart data) to model.json.

Run:  python train_model.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

data = pd.read_csv("data/diabetes1.csv")
raw = data.copy()

# --- Step 1: replace impossible zeros with the column median (as in the notebook)
medians = {}
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    medians[col] = float(np.median(data[col]))
    data[col] = data[col].replace(0, medians[col])

# --- Step 2: min-max scale everything except Pregnancies and Outcome
scaled_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",
               "BMI", "DiabetesPedigreeFunction", "Age"]
sc = MinMaxScaler()
scaled = pd.DataFrame(sc.fit_transform(data[scaled_cols]), columns=scaled_cols)
final_df = pd.concat([scaled, data[["Pregnancies", "Outcome"]]], axis=1)

# --- Step 3: split + fit
X = final_df.iloc[:, :-1]
y = final_df.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# --- Step 4: evaluate
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
fpr, tpr, _ = roc_curve(y_test, prob)

features = list(X.columns)
model = {
    "features": features,
    "coef": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
    "scaled": scaled_cols,
    "data_min": dict(zip(scaled_cols, sc.data_min_.tolist())),
    "data_max": dict(zip(scaled_cols, sc.data_max_.tolist())),
    "medians": medians,
    "feature_mean_scaled": {f: float(X[f].mean()) for f in features},
    "pregnancies_range": [int(raw.Pregnancies.min()), int(raw.Pregnancies.max())],
    "metrics": {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob),
        "train_accuracy": accuracy_score(y_train, clf.predict(X_train)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    },
    "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
    "dataset": {
        "rows": int(len(raw)),
        "positive": int(raw.Outcome.sum()),
        "negative": int((raw.Outcome == 0).sum()),
        "zero_counts": {c: int((raw[c] == 0).sum()) for c in
                        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]},
    },
}

# Glucose histogram by outcome (cleaned data) for the "data" chart
bins = list(range(40, 210, 10))
hist = {}
for o in (0, 1):
    counts, _ = np.histogram(data.loc[data.Outcome == o, "Glucose"], bins=bins)
    hist[str(o)] = counts.tolist()
model["glucose_hist"] = {"bins": bins, "counts": hist}

# Diabetic rate by age band & by BMI band
def rate(series, edges, labels):
    cut = pd.cut(series, edges, labels=labels, right=False)
    g = data.groupby(cut, observed=True)["Outcome"].agg(["mean", "count"])
    return [{"label": str(i), "rate": float(r["mean"]), "n": int(r["count"])} for i, r in g.iterrows()]
model["age_rates"] = rate(data.Age, [20, 30, 40, 50, 60, 90], ["21–29", "30–39", "40–49", "50–59", "60+"])
model["bmi_rates"] = rate(data.BMI, [0, 25, 30, 35, 40, 100], ["<25", "25–29.9", "30–34.9", "35–39.9", "40+"])

# A few real patients from the dataset as one-click examples
full_prob = clf.predict_proba(X)[:, 1]
cand = pd.DataFrame({"p": full_prob, "y": data.Outcome, "i": data.index})
def pick(df):
    r = raw.loc[int(df.i)]
    return {c: float(r[c]) for c in raw.columns if c != "Outcome"} | {"outcome": int(r["Outcome"]), "model_prob": float(df.p)}
low = cand[(cand.y == 0) & (cand.p < 0.08)].sort_values("p").iloc[3]
high = cand[(cand.y == 1) & (cand.p > 0.85)].sort_values("p").iloc[0]
mid = cand[(cand.p > 0.45) & (cand.p < 0.55)].iloc[0]
model["examples"] = {"low": pick(low), "mid": pick(mid), "high": pick(high)}

# Typical (median) patient, used as the form's starting point
model["defaults"] = {c: float(np.median(data[c])) for c in raw.columns if c != "Outcome"}

with open("model.json", "w") as f:
    json.dump(model, f, indent=2)

print({k: round(v, 3) for k, v in model["metrics"].items() if isinstance(v, float)})
print(model["metrics"]["confusion"])
print(dict(zip(features, np.round(model["coef"], 3))), round(model["intercept"], 3))
print(model["examples"])
