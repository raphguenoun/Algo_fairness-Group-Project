CSV_PATH = "/mnt/c/Users/lenovo/Downloads/dataproject2025 (2).csv"
# === Lightweight Step-1 Surrogates for DP (fast & interpretable) ===
# - Surrogate A: Shallow Decision Tree (regresses DP)
# - Surrogate B: Logistic Regression (classifies median-split DP)
# ================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr

# --------------------
# 0) Load data (with optional fast subsample)
# --------------------
dp_col = "Predicted probabilities"
MAX_ROWS = 120_000     # set lower (e.g., 40_000) if your machine struggles
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)

if dp_col not in df.columns:
    raise ValueError(f"'{dp_col}' not found. Columns available: {list(df.columns)}")

if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

# --------------------
# 1) Drop leakage and non-feature identifiers
# --------------------
leak_patterns = ["predic", "probab", "dp", "pd", "default", "risk", "score", "target"]
leaky_cols = [c for c in df.columns if any(p in c.lower() for p in leak_patterns)]
non_feature_cols = ["issue_d", "zip_code"]  # feel free to add obvious IDs/dates here
drop_cols = list(set(leaky_cols + non_feature_cols))
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
y_dp = df[dp_col].astype(float).clip(0, 1)

# Clean interest rate like "13.5%"
if "int_rate" in X.columns and X["int_rate"].dtype == "object":
    X["int_rate"] = (
        X["int_rate"].astype(str).str.strip().str.rstrip("%").replace({"": None}).astype(float)
    )

# --------------------
# 2) Type inference & tame high-cardinality categoricals
# --------------------
num_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all = X.select_dtypes(include=["object", "category"]).columns.tolist()

TOP_K_CAT = 12  # much smaller than before for speed & clarity
for c in cat_cols_all:
    X[c] = X[c].astype("object").fillna("Missing")
    vc = X[c].value_counts(dropna=False)
    if len(vc) > TOP_K_CAT:
        keep = set(vc.nlargest(TOP_K_CAT).index)
        X[c] = np.where(X[c].isin(keep), X[c], "Other")

# Prefer sub_grade over grade if both exist
if "grade" in X.columns and "sub_grade" in X.columns:
    X = X.drop(columns=["grade"])
    num_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = X.select_dtypes(include=["object", "category"]).columns.tolist()

# --------------------
# 3) Lightweight feature screening (transparent & fast)
# --------------------
# Score numeric features by |Spearman(y, x)|
num_scores = []
for c in num_cols_all:
    s = X[c]
    if s.notna().sum() > 10 and s.var() > 0:
        try:
            corr = abs(spearmanr(s.fillna(s.median()), y_dp).correlation)
        except Exception:
            corr = 0.0
        if np.isnan(corr):
            corr = 0.0
        num_scores.append((c, corr))
num_scores.sort(key=lambda t: t[1], reverse=True)

# Score categoricals by |Spearman(y, mean-encoded(x))|
cat_scores = []
for c in cat_cols_all:
    s = X[c].astype("object").fillna("Missing")
    means = y_dp.groupby(s).mean()
    enc = s.map(means).astype(float)
    try:
        corr = abs(spearmanr(enc, y_dp).correlation)
    except Exception:
        corr = 0.0
    if np.isnan(corr):
        corr = 0.0
    cat_scores.append((c, corr))
cat_scores.sort(key=lambda t: t[1], reverse=True)

TOP_NUM = 8
TOP_CAT = 8
num_cols = [c for c, _ in num_scores[:TOP_NUM]]
cat_cols = [c for c, _ in cat_scores[:TOP_CAT]]

print(f"Selected numeric ({len(num_cols)}): {num_cols}")
print(f"Selected categorical ({len(cat_cols)}): {cat_cols}")

# --------------------
# 4) Surrogate A: Shallow Decision Tree (regresses DP)
#    - Ordinal encode categories (fine for trees, minimal overhead)
# --------------------
preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)

tree = DecisionTreeRegressor(
    max_depth=3,           # very interpretable
    min_samples_leaf=150,  # increase if your dataset is huge; keeps splits stable
    random_state=RANDOM_STATE,
)

pipe_tree = Pipeline(steps=[("prep", preprocess_tree), ("tree", tree)])

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X[num_cols + cat_cols], y_dp, test_size=0.2, random_state=RANDOM_STATE
)

pipe_tree.fit(X_train_t, y_train_t)
dp_pred = pipe_tree.predict(X_test_t)

print("\n==== Surrogate A: Decision Tree (regression on DP) ====")
print(f"R^2: {r2_score(y_test_t, dp_pred):.4f}")
print(f"Spearman(tree_pred, DP): {spearmanr(dp_pred, y_test_t).correlation:.4f}")

# Human-readable rules (need feature names *after* preprocessing)
# Reconstruct names (ordinal encoder keeps column order)
tree_feature_names = num_cols + cat_cols
rules = export_text(pipe_tree.named_steps["tree"], feature_names=tree_feature_names, max_depth=3)
print("\nTree rules (depth<=3):\n")
print("\n".join(rules.splitlines()[:80]))

# --------------------
# 5) Surrogate B: Logistic Regression (median-split DP)
#    - Small OHE + L1 (liblinear) = sparse & readable coefficients
# --------------------
y_cls = (y_dp >= y_dp.median()).astype(int)

num_pipe_cls = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),  # plays well with sparse union
])

def make_ohe_drop_first(dense=False):
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=not dense)

cat_pipe_cls = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", make_ohe_drop_first(dense=False)),
])

preprocess_cls = ColumnTransformer(
    transformers=[
        ("num", num_pipe_cls, num_cols),
        ("cat", cat_pipe_cls, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

logit = LogisticRegression(
    penalty="l1",
    solver="liblinear",   # fast on small sparse problems
    C=0.7,
    max_iter=2000,
)

pipe_logit = Pipeline(steps=[("prep", preprocess_cls), ("clf", logit)])

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X[num_cols + cat_cols], y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
)

pipe_logit.fit(X_train_l, y_train_l)
proba_test = pipe_logit.predict_proba(X_test_l)[:, 1]

auc = roc_auc_score(y_test_l, proba_test)
spearman_cls = spearmanr(proba_test, y_dp.loc[y_test_l.index]).correlation

print("\n==== Surrogate B: Logistic (classification on median-split DP) ====")
print(f"ROC-AUC vs median-split DP: {auc:.4f}")
print(f"Spearman(surrogate_proba, DP): {spearman_cls:.4f}")

# Coefficients (readable)
def get_feature_names_from_ct(ct: ColumnTransformer):
    try:
        return ct.get_feature_names_out()
    except Exception:
        # Fallback (rarely needed here)
        names = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps"):
                steps = list(trans.named_steps.values())
                last = None
                for step in reversed(steps):
                    if hasattr(step, "get_feature_names_out"):
                        last = step; break
                if last is not None:
                    names.extend(last.get_feature_names_out(cols))
                else:
                    names.extend(cols)
            elif hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return np.array(names, dtype=object)

feat_names = get_feature_names_from_ct(pipe_logit.named_steps["prep"])
coefs = pipe_logit.named_steps["clf"].coef_.ravel()
k = min(len(feat_names), len(coefs))

coef_df = pd.DataFrame({
    "feature": feat_names[:k],
    "coef": coefs[:k],
})
coef_df["odds_ratio"] = np.exp(coef_df["coef"])
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\nTop 20 features by |coef|:")
print(coef_df.head(20)[["feature", "coef", "odds_ratio"]].to_string(index=False))

# Save results and push to GitHub
coef_df.to_csv("logit_top_coef.csv", index=False)
with open("tree_rules.txt", "w", encoding="utf-8") as f:
    f.write(rules)
