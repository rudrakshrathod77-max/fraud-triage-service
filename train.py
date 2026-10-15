import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import add_features, clean_transactions

RAW_PATH = "transactions_raw.csv"
EXCLUDE = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]
SEED = 42
GRID = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
THRESHOLDS = [round(0.1 * k, 1) for k in range(1, 10)]


def load_table():
    return add_features(clean_transactions(pd.read_csv(RAW_PATH)))


def make_xy(table):
    X = pd.get_dummies(table.drop(columns=EXCLUDE), dtype=int)
    return X, table["is_fraud"]


def make_splits(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    median = float(X_tr["distance_from_home_km"].median())   # training split only
    filled = []
    for d in (X_tr, X_va, X_te):
        d = d.copy()
        d["distance_from_home_km"] = d["distance_from_home_km"].fillna(median)
        filled.append(d)
    X_tr, X_va, X_te = filled
    return {"X_tr": X_tr, "X_va": X_va, "X_te": X_te,
            "y_tr": y_tr, "y_va": y_va, "y_te": y_te, "median": median}


def new_forest(**params):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=SEED, **params)


def new_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_forest(X_tr, y_tr):
    grid = GridSearchCV(new_forest(), GRID, scoring="f1", cv=new_cv(), n_jobs=-1)
    grid.fit(X_tr, y_tr)
    return grid


def pick_threshold(y_va, proba):
    best = 0.1
    for t in THRESHOLDS:
        if recall_score(y_va, proba >= t, zero_division=0) >= 0.65:
            best = t
    return best


def build_final_model():
    X, y = make_xy(load_table())
    s = make_splits(X, y)
    grid = tune_forest(s["X_tr"], s["y_tr"])
    model = grid.best_estimator_
    proba = model.predict_proba(s["X_va"])[:, 1]
    return {"model": model, "columns": list(X.columns),
            "median": s["median"], "threshold": pick_threshold(s["y_va"], proba)}


def prf(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def lr_pipeline(**kw):
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, **kw))])


if __name__ == "__main__":
    table = load_table()
    X, y = make_xy(table)
    print("input columns:", len(X.columns))
    print(list(X.columns))

    # 1. split
    s = make_splits(X, y)
    X_tr, X_va, X_te = s["X_tr"], s["X_va"], s["X_te"]
    y_tr, y_va, y_te = s["y_tr"], s["y_va"], s["y_te"]
    print("\n== Split ==")
    print("sizes train/val/test:", len(X_tr), len(X_va), len(X_te))
    print("fraud train/val/test:", int(y_tr.sum()), int(y_va.sum()), int(y_te.sum()))
    print("training median distance_from_home_km:", round(s["median"], 2))

    # 2. three imbalance treatments (validation, threshold 0.5)
    plain = lr_pipeline().fit(X_tr, y_tr)
    balanced = lr_pipeline(class_weight="balanced").fit(X_tr, y_tr)
    train_df = X_tr.copy()
    train_df["is_fraud"] = y_tr
    non_fraud = train_df[train_df["is_fraud"] == 0]
    fraud_up = train_df[train_df["is_fraud"] == 1].sample(
        n=len(non_fraud), replace=True, random_state=SEED)
    over = pd.concat([non_fraud, fraud_up])
    oversampled = lr_pipeline().fit(over.drop(columns="is_fraud"), over["is_fraud"])
    rows = {"plain": plain, "balanced": balanced, "oversampled": oversampled}
    table_imb = pd.DataFrame({name: prf(y_va, m.predict_proba(X_va)[:, 1] >= 0.5)
                              for name, m in rows.items()}).T
    print("\n== Imbalance treatments (validation, threshold 0.5) ==")
    print(table_imb)
    print("over-sampled training rows:", len(over),
          "| fraud:", int(over["is_fraud"].sum()),
          "| non-fraud:", int((over["is_fraud"] == 0).sum()))

    # 3. cross-validation and tuning (training split only)
    cv = new_cv()
    lr_cv = cross_val_score(lr_pipeline(class_weight="balanced"), X_tr, y_tr, cv=cv, scoring="f1")
    rf_cv = cross_val_score(new_forest(), X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
    print("\n== Cross-validation F1 (5 folds, training split) ==")
    print(f"balanced logistic regression: mean {lr_cv.mean():.3f}, std {lr_cv.std():.3f}")
    print(f"untuned random forest:        mean {rf_cv.mean():.3f}, std {rf_cv.std():.3f}")
    grid = tune_forest(X_tr, y_tr)
    best = grid.best_estimator_
    best_std = grid.cv_results_["std_test_score"][grid.best_index_]
    print("grid combinations:", len(grid.cv_results_["params"]))
    print("best params:", grid.best_params_)
    print(f"best mean CV F1: {grid.best_score_:.3f} (std {best_std:.3f})")

    # 4. threshold on validation
    proba_va = best.predict_proba(X_va)[:, 1]
    thr_rows = {}
    for t in THRESHOLDS:
        r = prf(y_va, proba_va >= t)
        thr_rows[t] = {"precision": r["precision"], "recall": r["recall"],
                       "f1": r["f1"], "flagged": int((proba_va >= t).sum())}
    print("\n== Validation thresholds (tuned forest) ==")
    print(pd.DataFrame(thr_rows).T)
    t_star = pick_threshold(y_va, proba_va)
    print("t_star:", t_star)

    # 5. one final test evaluation
    pred_te = best.predict_proba(X_te)[:, 1] >= t_star
    test = prf(y_te, pred_te)
    print("\n== Test at t_star (evaluated once) ==")
    print(test, "| flagged:", int(pred_te.sum()))
    print("legit payments held per fraud caught:", round(test["FP"] / test["TP"], 2))
    print("frauds missed:", test["FN"])

    # 6. leakage experiment
    X_leak = X.copy()
    X_leak["chargeback_filed"] = table["chargeback_filed"].values   # 27th, last column
    sl = make_splits(X_leak, y)
    leaky = new_forest(**grid.best_params_).fit(sl["X_tr"], sl["y_tr"])
    clean_f1 = f1_score(y_va, proba_va >= 0.5)
    leaky_f1 = f1_score(sl["y_va"], leaky.predict_proba(sl["X_va"])[:, 1] >= 0.5)
    print("\n== Leakage experiment (validation F1 at 0.5) ==")
    print(f"clean tuned forest: {clean_f1:.3f} | leaky forest: {leaky_f1:.3f}")

    # 7. feature importance
    imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n== Top 5 features ==")
    print(imp.head(5).round(4))