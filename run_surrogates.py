# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from scipy.stats import spearmanr


def make_ohe_drop_first(dense: bool):
    """Version-safe OneHotEncoder with drop='first'."""
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=not dense)


def get_feature_names_from_ct(ct: ColumnTransformer):
    """Robust feature name extraction from ColumnTransformer (works with Pipelines)."""
    try:
        return ct.get_feature_names_out()
    except Exception:
        pass
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "named_steps"):
            steps = list(trans.named_steps.values())
            last = None
            for step in reversed(steps):
                if hasattr(step, "get_feature_names_out"):
                    last = step
                    break
            if last is not None:
                out = last.get_feature_names_out(cols if isinstance(cols, list) else [cols])
                names.extend(out)
            else:
                names.extend(cols if isinstance(cols, list) else [cols])
        elif hasattr(trans, "get_feature_names_out"):
            out = trans.get_feature_names_out(cols if isinstance(cols, list) else [cols])
            names.extend(out)
        else:
            names.extend(cols if isinstance(cols, list) else [cols])
    return np.array(names, dtype=object)


def main():
    parser = argparse.ArgumentParser(description="Surrogate models for black-box DP.")
    parser.add_argument("--csv", required=True, help="Path to your CSV (inside container).")
    parser.add_argument("--dp-col", default="Predicted probabilities", help="DP column name.")
    parser.add_argument("--topk", type=int, default=30, help="Top-K categories to keep per column.")
    parser.add_argument("--depth", type=int, default=4, help="Max depth for decision tree.")
    parser.add_argument("--leaf", type=int, default=100, help="Min samples per leaf for tree.")
    parser.add_argument("--logitC", type=float, default=0.5, help="C for L1 logistic.")
    parser.add_argument("--drop-grade", action="store_true",
                        help="If both grade & sub_grade exist, drop grade (keep sub_grade).")
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    # 1) Select DP & drop leakage
    dp_col = args.dp_col
    if dp_col not in df.columns:
        raise ValueError(f"'{dp_col}' not found. Columns: {list(df.columns)}")

    leak_patterns = ["predic", "probab", "dp", "pd", "default", "risk", "score", "target"]
    leaky_cols = [c for c in df.columns if any(p in c.lower() for p in leak_patterns)]
    non_feature_cols = ["issue_d", "zip_code"]
    drop_cols = list(set(leaky_cols + non_feature_cols))
    print("Dropping potential leakage/non-feature columns:", drop_cols)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y_dp = df[dp_col].astype(float).clip(0, 1)

    # Optional: clean interest rate if it's like "13.5%"
    if "int_rate" in X.columns and X["int_rate"].dtype == "object":
        X["int_rate"] = (
            X["int_rate"].astype(str).str.strip().str.rstrip("%").replace({"": None}).astype(float)
        )

    # 2) Types & cap high-cardinality
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    TOP_K = args.topk
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Missing")
        vc = X[c].value_counts(dropna=False)
        if len(vc) > TOP_K:
            keep = vc.nlargest(TOP_K).index
            X[c] = X[c].where(X[c].isin(keep), "Other")

    # Optional redundancy cut: keep sub_grade, drop grade if both exist
    if args.drop_grade and ("grade" in X.columns) and ("sub_grade" in X.columns):
        X = X.drop(columns=["grade"])
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 3) Logistic surrogate (classification on median split)
    y_cls = (y_dp >= y_dp.median()).astype(int)

    num_pipe_cls = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
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
        solver="saga",
        C=args.logitC,
        max_iter=4000,
        n_jobs=-1,
    )

    pipe_logit = Pipeline(steps=[("prep", preprocess_cls), ("clf", logit)])

    X_train, X_test, y_train, y_test, dp_train, dp_test = train_test_split(
        X, y_cls, y_dp, test_size=0.2, random_state=42, stratify=y_cls
    )
    pipe_logit.fit(X_train, y_train)
    proba_test = pipe_logit.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba_test)
    spearman_cls = spearmanr(proba_test, dp_test).correlation

    print("\n==== Logistic Surrogate (classification) ====")
    print(f"ROC-AUC vs median-split DP: {auc:.4f}")
    print(f"Spearman corr(surrogate_proba, blackbox_DP): {spearman_cls:.4f}")

    feat_names_cls = get_feature_names_from_ct(pipe_logit.named_steps["prep"])
    coefs = pipe_logit.named_steps["clf"].coef_.ravel()
    k = min(len(feat_names_cls), len(coefs))
    coef_df = pd.DataFrame({"feature": feat_names_cls[:k], "coef": coefs[:k]})
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    print("\nTop 20 most influential features (by |coef|):")
    print(coef_df.head(20)[["feature", "coef", "odds_ratio"]].to_string(index=False))

    # 4) Decision Tree surrogate (regression on DP)
    num_pipe_reg = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe_reg = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe_drop_first(dense=True)),  # dense for readable thresholds
    ])
    preprocess_reg = ColumnTransformer(
        transformers=[
            ("num", num_pipe_reg, num_cols),
            ("cat", cat_pipe_reg, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    tree = DecisionTreeRegressor(
        max_depth=args.depth,
        min_samples_leaf=args.leaf,
        random_state=42,
    )
    pipe_tree = Pipeline(steps=[("prep", preprocess_reg), ("tree", tree)])

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_dp, test_size=0.2, random_state=42
    )
    pipe_tree.fit(X_train2, y_train2)
    dp_pred = pipe_tree.predict(X_test2)

    r2 = r2_score(y_test2, dp_pred)
    spearman_tree = spearmanr(dp_pred, y_test2).correlation

    print("\n==== Decision Tree Surrogate (regression) ====")
    print(f"R^2 score: {r2:.4f}")
    print(f"Spearman corr(tree_pred, blackbox_DP): {spearman_tree:.4f}")

    # Tree rules
    ohe = pipe_tree.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    num_names = num_cols
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    tree_feat_names = num_names + cat_names
    rules = export_text(pipe_tree.named_steps["tree"], feature_names=tree_feat_names, max_depth=args.depth)
    print("\nTree rules (depth<=%d):\n" % args.depth)
    print("\n".join(rules.splitlines()[:80]))

    # Importances
    imp = pipe_tree.named_steps["tree"].feature_importances_
    imp_df = pd.DataFrame({"feature": tree_feat_names, "importance": imp}).sort_values("importance", ascending=False)
    print("\nTop 20 important features in tree:")
    print(imp_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
