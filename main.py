import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


# =====================================================
# 1. Read Excel file
# =====================================================

file_path = Path(__file__).parent / "pone.0124893.s001.xlsx"
print("FILE:", file_path.resolve())
print("SHEETS:", pd.ExcelFile(file_path).sheet_names)

df = pd.read_excel(file_path, sheet_name="Sheet1")

print(df.head())


# =====================================================
# 2. Select important columns
# =====================================================

selected_columns = [
    "Lab number",
    "Age (y)",
    "Gender",
    "Previous chemotherapy",
    "Treatment regime",
    "DPYD rs3918290; IVS14+1 G>A; Exon-skipping SNP in intron 14",
    "DPYD 1236G>A; E412E",
    "DPYD rs67376798; 2846A>T; D949V",
    "DPYD c1129-5923C>G",
    "TYMP rs11479; 1412C>T; S471L",
    "Grade 3, 4 or 5 toxicity events confirmed"
]
print(df.columns)
df = df[selected_columns].copy()


# =====================================================
# 3. Rename columns
# =====================================================

df = df.rename(columns={
    "Lab number": "patient_id",
    "Age (y)": "age",
    "Gender": "gender",
    "Previous chemotherapy": "previous_chemo",
    "Treatment regime": "treatment",
    "DPYD rs3918290; IVS14+1 G>A; Exon-skipping SNP in intron 14": "dpyd_rs3918290",
    "DPYD 1236G>A; E412E": "dpyd_1236GA",
    "DPYD rs67376798; 2846A>T; D949V": "dpyd_2846AT",
    "DPYD c1129-5923C>G": "dpyd_c1129",
    "TYMP rs11479; 1412C>T; S471L": "tymp_rs11479",
    "Grade 3, 4 or 5 toxicity events confirmed": "toxicity"
})


# =====================================================
# 4. Create target
# =====================================================

df["target"] = (
    df["toxicity"] == "Grade 3,4 or 5 event"
).astype(int)

print("\n--- Target ---")
print(df["target"].value_counts())


# =====================================================
# 5. Define X and y
# =====================================================

X = df[
    [
        "age",
        "gender",
        "previous_chemo",
        "treatment",
        "dpyd_rs3918290",
        "dpyd_1236GA",
        "dpyd_2846AT",
        "dpyd_c1129",
        "tymp_rs11479"
    ]
]

y = df["target"]

print("\nX:")
print(X.head())
print("\ny:")
print(y.head())


# =====================================================
# 6. Convert text into numbers
# =====================================================

X = pd.get_dummies(X)

print("\n--- X after conversion ---")
print(X.head())
print("\nX shape:")
print(X.shape)


# =====================================================
# 7. Split data
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Train/Test ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nData preparation completed!")


model = LogisticRegression(max_iter=1000)
print("\nModel created!")


# =====================================================
# 8. Train the model
# =====================================================

model.fit(X_train, y_train)
print("\nModel training completed!")


# =====================================================
# 9. Make predictions
# =====================================================

y_pred = model.predict(X_test)
print("\n--- Predictions ---")
print(y_pred)


# =====================================================
# 10. Evaluate the model properly
# =====================================================

y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- ROC AUC Score ---")
print(roc_auc_score(y_test, y_proba))


# =====================================================
# 11. Try a balanced model (fix imbalance issue)
# =====================================================

model_balanced = LogisticRegression(max_iter=1000, class_weight="balanced")
model_balanced.fit(X_train, y_train)

y_pred_balanced = model_balanced.predict(X_test)
y_proba_balanced = model_balanced.predict_proba(X_test)[:, 1]

print("\n--- Balanced Model: Classification Report ---")
print(classification_report(y_test, y_pred_balanced))
print("\n--- Balanced Model: Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_balanced))
print("\n--- Balanced Model: ROC AUC Score ---")
print(roc_auc_score(y_test, y_proba_balanced))


# =====================================================
# 12. Check which features matter most
# =====================================================

coefs = pd.Series(model_balanced.coef_[0], index=X.columns)
print("\n--- Feature Importance (Balanced Model) ---")
print(coefs.sort_values(key=abs, ascending=False))


# =====================================================
# 13. Cross-validation for a more reliable estimate
# =====================================================

cv_scores = cross_val_score(model_balanced, X, y, cv=5, scoring="roc_auc")
print("\n--- 5-Fold Cross Validation (ROC AUC) ---")
print(cv_scores)
print("Mean ROC AUC:", cv_scores.mean())


# =====================================================
# 14. Check toxicity rate by genotype
# =====================================================

for col in ["dpyd_rs3918290", "dpyd_1236GA", "dpyd_2846AT", "dpyd_c1129", "tymp_rs11479"]:
    print(f"\n--- Toxicity by {col} ---")
    print(pd.crosstab(df[col], df["target"], normalize="index"))


# =====================================================
# 15. Clinical-only vs Clinical + Genetic model
# =====================================================

clinical_features = ["age", "gender", "previous_chemo", "treatment"]
genetic_features = ["dpyd_rs3918290", "dpyd_1236GA", "dpyd_2846AT", "dpyd_c1129", "tymp_rs11479"]

X_clinical = pd.get_dummies(df[clinical_features])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clinical, y, test_size=0.2, random_state=42, stratify=y
)
clinical_model = LogisticRegression(max_iter=1000, class_weight="balanced")
clinical_model.fit(X_train_c, y_train_c)
clinical_auc = roc_auc_score(y_test_c, clinical_model.predict_proba(X_test_c)[:, 1])

X_all = pd.get_dummies(df[clinical_features + genetic_features])
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)
genetic_model = LogisticRegression(max_iter=1000, class_weight="balanced")
genetic_model.fit(X_train_g, y_train_g)
genetic_auc = roc_auc_score(y_test_g, genetic_model.predict_proba(X_test_g)[:, 1])

print("\n--- Model comparison ---")
print("Clinical-only ROC AUC:", clinical_auc)
print("Clinical + Genetic ROC AUC:", genetic_auc)
print("Difference:", genetic_auc - clinical_auc)


# =====================================================
# 16. Evaluate model with Sensitivity and Specificity
# =====================================================

y_pred = model_balanced.predict(X_test)
y_proba = model_balanced.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- Step 16: Model Performance ---")
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)
print("F1 Score:", f1)
print("ROC AUC:", auc)


# =====================================================
# 17. Try different Machine Learning models
# =====================================================

rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print("\n--- Step 17: Random Forest ---")
print("ROC AUC:", rf_auc)
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# =====================================================
# 18. Compare Logistic Regression and Random Forest
# =====================================================

print("\n--- Step 18: Model Comparison ---")
print("Logistic Regression AUC:", auc)
print("Random Forest AUC:", rf_auc)

if rf_auc > auc:
    print("Random Forest performed better.")
else:
    print("Logistic Regression performed better.")


# =====================================================
# 19. Feature importance from Random Forest
# =====================================================

rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importance = rf_importance.sort_values(ascending=False)

print("\n--- Step 19: Random Forest Feature Importance ---")
print(rf_importance)


# =====================================================
# 20. Predict toxicity risk for a new patient
# =====================================================

print("\n--- Step 20: New Patient Prediction ---")

new_patient = pd.DataFrame({
    "age": [65],
    "gender": ["Female"],
    "previous_chemo": ["No"],
    "treatment": ["Combination"],
    "dpyd_rs3918290": ["GG"],
    "dpyd_1236GA": ["GG"],
    "dpyd_2846AT": ["AA"],
    "dpyd_c1129": ["CC"],
    "tymp_rs11479": ["CT"]
})

new_patient = pd.get_dummies(new_patient)
new_patient = new_patient.reindex(columns=X.columns, fill_value=False)

new_patient_probability = model_balanced.predict_proba(new_patient)[:, 1][0]
new_patient_prediction = model_balanced.predict(new_patient)[0]

print("Probability of severe toxicity:", new_patient_probability)
print("Predicted class:", new_patient_prediction)

if new_patient_prediction == 1:
    print("Prediction: HIGHER toxicity risk")
else:
    print("Prediction: LOWER toxicity risk")


# =====================================================
# 21. Threshold Analysis
# =====================================================
# پیش‌فرض predict() این است که اگر احتمال > 0.5 باشد، کلاس را ۱ در نظر می‌گیرد.
# ولی این عدد ۰.۵ هیچ قانون ثابتی نیست؛ در مسائل پزشکی معمولاً باید آستانه
# (threshold) را پایین‌تر بیاوریم تا بیماران پرخطر بیشتری شناسایی شوند،
# حتی اگر هشدارهای اشتباه (False Positive) بیشتری هم داشته باشیم.

print("\n--- Step 21: Threshold Analysis ---")

thresholds_to_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

for t in thresholds_to_test:
    y_pred_t = (y_proba_balanced >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)

    threshold_results.append({
        "threshold": t,
        "sensitivity": round(sens_t, 3),
        "specificity": round(spec_t, 3),
        "precision": round(prec_t, 3),
        "TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df)


# =====================================================
# 22. ROC Curve
# =====================================================
# منحنی ROC نشان می‌دهد در آستانه‌های مختلف، نسبت True Positive Rate
# (حساسیت) به False Positive Rate چگونه تغییر می‌کند.
# هرچه منحنی به گوشه بالا-چپ نزدیک‌تر باشد، مدل بهتر است.

print("\n--- Step 22: ROC Curve ---")

fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba_balanced)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Balanced LR (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

print("ROC curve saved as roc_curve.png")


# =====================================================
# 23. Feature Importance Visualization
# =====================================================
# همون rf_importance گام ۱۹ رو این بار به‌جای print، به‌صورت نمودار میله‌ای
# رسم می‌کنیم تا خواندنش راحت‌تر باشه (مخصوصاً برای گزارش یا رزومه).

print("\n--- Step 23: Feature Importance Plot ---")

top_features = rf_importance.head(10)

plt.figure(figsize=(7, 5))
top_features.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("Feature importance plot saved as feature_importance.png")


# =====================================================
# 24. Precision-Recall Curve
# =====================================================
# چون کلاس toxicity (هدف ما) خیلی کمتر از کلاس دیگر است (imbalanced)،
# منحنی Precision-Recall معمولاً تصویر واقعی‌تری از عملکرد مدل نسبت به
# ROC Curve می‌دهد.

print("\n--- Step 24: Precision-Recall Curve ---")

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba_balanced)
avg_precision = average_precision_score(y_test, y_proba_balanced)

plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, label=f"Balanced LR (AP = {avg_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.close()

print("Average Precision:", avg_precision)
print("Precision-Recall curve saved as precision_recall_curve.png")


# =====================================================
# 25. Compare several ML models side by side
# =====================================================
# چند مدل مختلف را روی همون train/test split امتحان می‌کنیم تا ببینیم
# کدوم بهتر عمل می‌کنه. هدف این نیست که فقط "بهترین عدد" رو پیدا کنیم،
# بلکه یاد بگیریم چطور مدل‌ها رو منصفانه مقایسه کنیم.

print("\n--- Step 25: Compare Multiple Models ---")

models_to_compare = {
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

comparison_results = []

for name, m in models_to_compare.items():
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:, 1]
    pred = m.predict(X_test)
    model_auc = roc_auc_score(y_test, proba)
    model_recall = recall_score(y_test, pred)
    model_precision = precision_score(y_test, pred, zero_division=0)

    comparison_results.append({
        "model": name,
        "ROC_AUC": round(model_auc, 3),
        "recall_class1": round(model_recall, 3),
        "precision_class1": round(model_precision, 3),
    })

comparison_df = pd.DataFrame(comparison_results)
print(comparison_df)


# =====================================================
# 26. Cross-Validation for each model
# =====================================================
# یه تک train/test split می‌تونه گمراه‌کننده باشه (به‌خصوص با دیتاست کوچیک).
# این‌جا هر مدل رو با ۵-fold cross-validation روی کل داده تست می‌کنیم تا
# نتیجه‌ی پایدارتری بگیریم.

print("\n--- Step 26: Cross-Validation for Each Model ---")

cv_comparison = []

for name, m in models_to_compare.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    cv_comparison.append({
        "model": name,
        "CV_mean_AUC": round(scores.mean(), 3),
        "CV_std_AUC": round(scores.std(), 3),
    })

cv_comparison_df = pd.DataFrame(cv_comparison)
print(cv_comparison_df)


# =====================================================
# 27. Hyperparameter Tuning
# =====================================================
# GridSearchCV تمام ترکیب‌های ممکن از پارامترها رو امتحان می‌کنه و بهترین
# ترکیب رو بر اساس معیار انتخابی (اینجا roc_auc) پیدا می‌کنه.
# این کار روی دیتاست کوچیک ما زمان زیادی نمی‌بره.

print("\n--- Step 27: Hyperparameter Tuning (Random Forest) ---")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 3, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best CV ROC AUC:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_


# =====================================================
# 28. Check for Data Leakage and Missing Values
# =====================================================
# قبل از اینکه به نتایج مدل زیادی اعتماد کنیم، باید مطمئن بشیم که هیچ
# اشتباهی در آماده‌سازی داده باعث نتیجه‌ی مصنوعی خوب یا بد نشده.

print("\n--- Step 28: Data Leakage & Missing Value Checks ---")

# ۱. چک کردن مقادیر گم‌شده در ستون‌های استفاده‌شده
print("\nMissing values per column:")
print(df[[
    "age", "gender", "previous_chemo", "treatment",
    "dpyd_rs3918290", "dpyd_1236GA", "dpyd_2846AT",
    "dpyd_c1129", "tymp_rs11479", "target"
]].isna().sum())

# ۲. مطمئن می‌شویم patient_id (که فقط یک شناسه است و معنای بالینی ندارد)
#    هرگز به‌عنوان یک ویژگی وارد X نشده باشد
print("\n'patient_id' in X columns?", any("patient_id" in c for c in X.columns))

# ۳. مطمئن می‌شویم بین train و test هیچ بیمار مشترکی وجود ندارد
train_index_set = set(X_train.index)
test_index_set = set(X_test.index)
overlap = train_index_set.intersection(test_index_set)
print("Number of overlapping patients between train and test:", len(overlap))

# ۴. توزیع کلاس هدف را در train و test مقایسه می‌کنیم
print("\nTarget distribution in y_train:")
print(y_train.value_counts(normalize=True))
print("\nTarget distribution in y_test:")
print(y_test.value_counts(normalize=True))


# =====================================================
# 29. Build a Reusable Prediction Function
# =====================================================
# به‌جای تکرار کد گام ۲۰ هر بار، یک تابع می‌سازیم که یک دیکشنری از
# اطلاعات بیمار می‌گیرد و ریسک سمیت را برمی‌گرداند.

print("\n--- Step 29: Reusable Prediction Function ---")


def predict_toxicity_risk(patient_info: dict, model=model_balanced, reference_columns=X.columns):
    """
    patient_info: دیکشنری با کلیدهایی مثل age, gender, previous_chemo,
                  treatment, dpyd_rs3918290, dpyd_1236GA, dpyd_2846AT,
                  dpyd_c1129, tymp_rs11479

    خروجی: دیکشنری شامل احتمال، کلاس پیش‌بینی‌شده و برچسب ریسک
    """
    patient_df = pd.DataFrame([patient_info])
    patient_df = pd.get_dummies(patient_df)
    patient_df = patient_df.reindex(columns=reference_columns, fill_value=False)

    probability = model.predict_proba(patient_df)[:, 1][0]
    prediction = model.predict(patient_df)[0]

    if probability < 0.3:
        risk_label = "Low"
    elif probability < 0.6:
        risk_label = "Intermediate"
    else:
        risk_label = "High"

    return {
        "probability": round(float(probability), 3),
        "predicted_class": int(prediction),
        "risk_label": risk_label,
    }


# تست تابع با همون بیمار نمونه گام ۲۰
example_patient = {
    "age": 65,
    "gender": "Female",
    "previous_chemo": "No",
    "treatment": "Combination",
    "dpyd_rs3918290": "GG",
    "dpyd_1236GA": "GG",
    "dpyd_2846AT": "AA",
    "dpyd_c1129": "CC",
    "tymp_rs11479": "CT",
}

print(predict_toxicity_risk(example_patient))


# =====================================================
# 30. Final Summary for the GitHub-Ready Project
# =====================================================
# این بخش فقط یک خلاصه نهایی از مهم‌ترین نتایج پروژه را چاپ می‌کند،
# همان چیزی که باید در بخش "Results" فایل README.md قرار بگیرد.

print("\n--- Step 30: Final Project Summary ---")

final_summary = pd.DataFrame([
    {"model": "Logistic Regression (baseline)", "test_AUC": round(roc_auc_score(y_test, y_proba), 3)},
    {"model": "Logistic Regression (balanced)", "test_AUC": round(auc, 3)},
    {"model": "Random Forest (balanced)", "test_AUC": round(rf_auc, 3)},
    {"model": "Random Forest (tuned)", "CV_AUC": round(grid_search.best_score_, 3)},
])

print(final_summary)
print("\nClinical-only vs Clinical+Genetic AUC difference:", round(genetic_auc - clinical_auc, 3))
print("\nAll plots saved: roc_curve.png, feature_importance.png, precision_recall_curve.png")

print("\n======================================")
print("PROJECT STEPS 1-30 COMPLETED")
print("======================================")
import sys
import time


def step_start(step_name):
    print(f"\n>>> STARTING: {step_name}", flush=True)
    return time.time()


def step_done(step_name, t0):
    print(f">>> FINISHED: {step_name} ({time.time() - t0:.2f}s)", flush=True)


# =====================================================
# 31. Statistical Analysis of Genetic Variants
# =====================================================
t0 = step_start("Step 31 - Chi-square/Fisher tests")
try:
    from scipy.stats import chi2_contingency, fisher_exact

    genetic_columns = [
        "dpyd_rs3918290", "dpyd_1236GA", "dpyd_2846AT",
        "dpyd_c1129", "tymp_rs11479"
    ]

    for col in genetic_columns:
        table = pd.crosstab(df[col], df["target"])
        print(f"\nVariant: {col}", flush=True)
        print(table, flush=True)

        if table.shape == (2, 2):
            odds_ratio, p_value = fisher_exact(table)
            print("Fisher's Exact Test | Odds Ratio:", odds_ratio, "| P-value:", p_value, flush=True)
        else:
            chi2, p_value, dof, expected = chi2_contingency(table)
            print("Chi-square Test | P-value:", p_value, flush=True)
except Exception as e:
    print(f"!!! STEP 31 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 31", t0)


# =====================================================
# 32. Calculate Odds Ratios for Logistic Regression
# =====================================================
t0 = step_start("Step 32 - Odds Ratios")
try:
    import numpy as np

    logistic_coefficients = pd.Series(model_balanced.coef_[0], index=X.columns)
    odds_ratios = np.exp(logistic_coefficients)

    odds_ratio_table = pd.DataFrame({
        "feature": logistic_coefficients.index,
        "coefficient": logistic_coefficients.values,
        "odds_ratio": odds_ratios.values
    }).sort_values("odds_ratio", ascending=False)

    print(odds_ratio_table, flush=True)
except Exception as e:
    print(f"!!! STEP 32 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 32", t0)


# =====================================================
# 33. Statistical Significance of Logistic Regression
# =====================================================
t0 = step_start("Step 33 - statsmodels Logit")
stats_result = None
try:
    import statsmodels.api as sm

    # drop_first=True avoids the dummy-variable trap / perfect multicollinearity
    X_for_stats = pd.get_dummies(
        df[["age", "gender", "previous_chemo", "treatment",
            "dpyd_rs3918290", "dpyd_1236GA", "dpyd_2846AT",
            "dpyd_c1129", "tymp_rs11479"]],
        drop_first=True
    )
    X_stats = sm.add_constant(X_for_stats.astype(float))

    print("Design matrix shape:", X_stats.shape, flush=True)
    print("Design matrix rank check (should equal number of columns):", flush=True)
    print(np.linalg.matrix_rank(X_stats.values), "vs", X_stats.shape[1], flush=True)

    stats_model = sm.Logit(y, X_stats)
    # maxiter caps runtime so it CANNOT hang forever; regularized fit handles separation
    stats_result = stats_model.fit_regularized(disp=True, alpha=0.1, maxiter=100)
    print(stats_result.summary(), flush=True)
except Exception as e:
    print(f"!!! STEP 33 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 33", t0)


# =====================================================
# 34. Confidence Intervals for Odds Ratios
# =====================================================
t0 = step_start("Step 34 - Confidence Intervals")
try:
    if stats_result is not None:
        params = stats_result.params
        # fit_regularized() result may not expose conf_int() the same way; guard for it
        if hasattr(stats_result, "conf_int"):
            conf = stats_result.conf_int()
            or_table = pd.DataFrame({
                "feature": params.index,
                "odds_ratio": np.exp(params.values),
                "CI_lower": np.exp(conf[0].values),
                "CI_upper": np.exp(conf[1].values),
            })
            print(or_table, flush=True)
        else:
            print("This statsmodels result type has no conf_int() (common for fit_regularized).", flush=True)
            print("Point estimates only:", flush=True)
            print(np.exp(params), flush=True)
    else:
        print("Skipped: Step 33 did not produce a valid stats_result.", flush=True)
except Exception as e:
    print(f"!!! STEP 34 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 34", t0)


# =====================================================
# 35. Calibration Analysis
# =====================================================
t0 = step_start("Step 35 - Calibration Analysis")
brier = None
prob_true, prob_pred = None, None
try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    prob_true, prob_pred = calibration_curve(
        y_test, y_proba_balanced, n_bins=5, strategy="quantile"
    )
    brier = brier_score_loss(y_test, y_proba_balanced)

    print("Brier Score:", brier, flush=True)
    print("Predicted probability:", prob_pred, flush=True)
    print("Observed frequency:", prob_true, flush=True)
except Exception as e:
    print(f"!!! STEP 35 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 35", t0)


# =====================================================
# 36. Calibration Plot
# =====================================================
t0 = step_start("Step 36 - Calibration Plot")
try:
    if prob_true is not None:
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="o", label="Balanced Logistic Regression")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title("Calibration Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("calibration_curve.png", dpi=150)
        plt.close()
        print("Calibration curve saved as calibration_curve.png", flush=True)
    else:
        print("Skipped: Step 35 did not produce calibration data.", flush=True)
except Exception as e:
    print(f"!!! STEP 36 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 36", t0)


# =====================================================
# 37. SHAP Explainability
# =====================================================
t0 = step_start("Step 37 - SHAP")
try:
    import shap

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(shap_values_plot, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved as shap_summary.png", flush=True)
except ImportError:
    print("SHAP is not installed. Install it with: pip install shap", flush=True)
except Exception as e:
    print(f"!!! STEP 37 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 37", t0)


# =====================================================
# 38. Compare Feature Importance Methods
# =====================================================
t0 = step_start("Step 38 - Feature Importance Comparison")
try:
    feature_comparison = pd.DataFrame({
        "feature": X.columns,
        "logistic_abs_coef": abs(model_balanced.coef_[0]),
        "random_forest_importance": rf_model.feature_importances_
    }).sort_values("random_forest_importance", ascending=False)

    print(feature_comparison.head(15), flush=True)
except Exception as e:
    print(f"!!! STEP 38 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 38", t0)


# =====================================================
# 39. Save Important Results
# =====================================================
t0 = step_start("Step 39 - Save CSV Results")
try:
    comparison_df.to_csv("model_comparison.csv", index=False)
    cv_comparison_df.to_csv("cross_validation_results.csv", index=False)
    rf_importance.to_csv("random_forest_feature_importance.csv", header=["importance"])
    odds_ratio_table.to_csv("logistic_odds_ratios.csv", index=False)
    threshold_df.to_csv("threshold_analysis.csv", index=False)
    print("Results saved successfully.", flush=True)
except Exception as e:
    print(f"!!! STEP 39 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 39", t0)


# =====================================================
# 40. Final Research Summary
# =====================================================
t0 = step_start("Step 40 - Final Summary")
try:
    print("\nDataset:", flush=True)
    print("Number of patients:", len(df), flush=True)
    print("Severe toxicity cases:", int(y.sum()), flush=True)

    print("\nModel Performance:", flush=True)
    print("Logistic Regression AUC:", round(auc, 3), flush=True)
    print("Random Forest AUC:", round(rf_auc, 3), flush=True)

    if brier is not None:
        print("\nCalibration - Brier Score:", round(brier, 3), flush=True)

    print("\n======================================", flush=True)
    print("PROJECT STEPS 1-40 COMPLETED", flush=True)
    print("======================================", flush=True)
except Exception as e:
    print(f"!!! STEP 40 FAILED: {type(e).__name__}: {e}", flush=True)
step_done("Step 40", t0)