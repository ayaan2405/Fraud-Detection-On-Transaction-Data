import os
import sys
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, precision_recall_curve, auc
    )
    from imblearn.over_sampling import SMOTE
except Exception as e:
    print("Missing packages. Install dependencies:")
    print("pip install -r requirements.txt")
    raise e

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def print_separator():
    print("="*80)

def evaluate_model(name, model, X_test, y_test, show_cm=True):
    pred = model.predict(X_test)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        prob = model.decision_function(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    rocauc = roc_auc_score(y_test, prob) if prob is not None else float("nan")

    print(f"--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {rocauc:.4f}")
    if show_cm:
        cm = confusion_matrix(y_test, pred)
        print("Confusion Matrix:")
        print(cm)
    print()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "rocauc": rocauc, "prob": prob, "pred": pred}

def plot_precision_recall(y_test, prob, title="Precision-Recall curve"):
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

DATA_FILE = "creditcard.csv"  
SMOTE_SAMPLING = 0.25         
RANDOM_STATE = 42

if not os.path.exists(DATA_FILE):
    print(f"ERROR: {DATA_FILE} not found in current directory: {os.getcwd()}")
    sys.exit(1)

print("Loading data...")
df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print_separator()

print("Class distribution:")
print(df['Class'].value_counts(normalize=True))
print("Counts:")
print(df['Class'].value_counts())
print_separator()

print("Sample rows:")
print(df.sample(5))
print_separator()

possible_id_cols = ['id', 'Id', 'ID', 'TransactionID', 'transaction_id']
for c in possible_id_cols:
    if c in df.columns:
        print(f"Dropping ID column: {c}")
        df = df.drop(columns=[c])

if df.isnull().any().any():
    print("Filling missing numeric values with median.")
    df = df.fillna(df.median())

TARGET = 'Class'
if TARGET not in df.columns:
    print(f"Target column '{TARGET}' not found. Columns available:\n{df.columns}")
    sys.exit(1)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print("One-hot encoding categorical columns:", cat_cols)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("Features shape after encoding:", X.shape)
print_separator()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train class distribution:", np.bincount(y_train)/len(y_train))
print("Test class distribution :", np.bincount(y_test)/len(y_test))
print_separator()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

print("Training baseline LogisticRegression (no balancing) ...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
res_lr = evaluate_model("LogisticRegression (no balance)", lr, X_test_scaled, y_test)

print("Training baseline RandomForest (no balancing) ...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)   
res_rf = evaluate_model("RandomForest (no balance)", rf, X_test, y_test)

print("Applying SMOTE on training data (only) ...")
sm = SMOTE(sampling_strategy=SMOTE_SAMPLING, random_state=RANDOM_STATE)
X_res_scaled, y_res = sm.fit_resample(X_train_scaled, y_train)
print("Resampled train shape (scaled):", X_res_scaled.shape)
print("Resampled class distribution:", np.bincount(y_res)/len(y_res))
print_separator()

print("Training LogisticRegression on SMOTE data ...")
lr_sm = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_sm.fit(X_res_scaled, y_res)
res_lr_sm = evaluate_model("LogisticRegression (SMOTE)", lr_sm, X_test_scaled, y_test)

print("Training RandomForest on SMOTE data (unscaled resample) ...")
X_res_unscaled, y_res_unscaled = SMOTE(sampling_strategy=SMOTE_SAMPLING, random_state=RANDOM_STATE).fit_resample(X_train, y_train)
rf_sm = RandomForestClassifier(
    n_estimators=120,
    max_depth=15,
    min_samples_split=3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_sm.fit(X_res_unscaled, y_res_unscaled)
res_rf_sm = evaluate_model("RandomForest (SMOTE)", rf_sm, X_test, y_test)

print("Training LogisticRegression with class_weight='balanced' ...")
lr_cw = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr_cw.fit(X_train_scaled, y_train)
res_lr_cw = evaluate_model("LogisticRegression (class_weight)", lr_cw, X_test_scaled, y_test)

print("Training RandomForest with class_weight='balanced' ...")
rf_cw = RandomForestClassifier(
    n_estimators=120,
    class_weight='balanced',
    max_depth=12,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_cw.fit(X_train, y_train)
res_rf_cw = evaluate_model("RandomForest (class_weight)", rf_cw, X_test, y_test)

if XGBOOST_AVAILABLE:
    print("Training XGBoost with scale_pos_weight ...")
    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos,
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb.fit(X_train, y_train)
    res_xgb = evaluate_model("XGBoost (scale_pos_weight)", xgb, X_test, y_test)
else:
    print("XGBoost not available. Skipping XGBoost section.")
print_separator()

best_model = lr_sm
prob = None
if hasattr(best_model, "predict_proba"):
    prob = best_model.predict_proba(X_test_scaled)[:, 1]
elif hasattr(best_model, "decision_function"):
    prob = best_model.decision_function(X_test_scaled)

if prob is not None:
    print("Plotting precision-recall curve for threshold tuning ...")
    plot_precision_recall(y_test, prob, title="PR curve (chosen model)")
    precisions, recalls, thresholds = precision_recall_curve(y_test, prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Best threshold by F1 ~ {best_threshold:.4f}")
    y_pred_thresh = (prob >= best_threshold).astype(int)
    print("Metrics at tuned threshold:")
    print("Precision:", precision_score(y_test, y_pred_thresh))
    print("Recall   :", recall_score(y_test, y_pred_thresh))
    print("F1       :", f1_score(y_test, y_pred_thresh))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_thresh))
else:
    print("No probability scores from chosen model; skip threshold tuning.")
print_separator()

print("Hyperparameter tuning for RandomForest (RandomizedSearchCV) ...")
from scipy.stats import randint as sp_randint

param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [8, 12, 15],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rs = RandomizedSearchCV(rf_base, param_distributions=param_dist, n_iter=8, scoring='recall', cv=2, random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
t0 = time.time()
rs.fit(X_res_unscaled, y_res_unscaled)
print("RandomizedSearchCV done in {:.1f}s".format(time.time()-t0))
print("Best params:", rs.best_params_)
best_rf_tuned = rs.best_estimator_
res_rf_tuned = evaluate_model("RandomForest (tuned on SMOTE)", best_rf_tuned, X_test, y_test)

print("Saving best models and artifacts ...")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(lr_sm, "logistic_smote.joblib")
joblib.dump(rf_sm, "rf_smote.joblib")
joblib.dump(rf_cw, "rf_classweight.joblib")
if XGBOOST_AVAILABLE:
    joblib.dump(xgb, "xgboost_model.joblib")
joblib.dump(best_rf_tuned, "rf_tuned.joblib")
print("Saved models to current directory.")
print_separator()
print("Example: inference on a single test sample (index 0) using Logistic (SMOTE) ...")
sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
pred_prob = lr_sm.predict_proba(sample_scaled)[0,1]
pred_label = lr_sm.predict(sample_scaled)[0]
print("Pred prob fraud:", pred_prob, "Pred label:", pred_label)
print_separator()

print("SUMMARY / RECOMMENDATIONS:")
print("- Prioritize recall for fraud detection while keeping precision acceptable.")
print("- Try both: SMOTE + tree models, and class_weight-based training.")
print("- Tune thresholds based on Precision-Recall curve for your operating point.")
print("- Use XGBoost/LightGBM/CatBoost for better speed and performance on tabular data.")
print("- Engineer transaction/time-based features for big improvements.")
print("Project complete.")
