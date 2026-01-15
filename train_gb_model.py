#!/usr/bin/env python3
# train_gb_model.py
#
# Train Gradient Boosting models using matches_with_elo.csv and save:
#   - model_gb.joblib                 (winner classifier)
#   - feature_columns_gb.csv          (features for winner model)
#   - model_games.joblib              (total games regressor, if total_games exists)
#   - feature_columns_games_gb.csv    (features for games model)
#
#     create labels by mirroring each match:
#   - row 1: winner vs loser  (y = 1)
#   - row 2: loser vs winner  (y = 0, with diff features sign-flipped)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import joblib

# 1. Load base data 
df_raw = pd.read_csv("matches_with_elo.csv")
print(f"Loaded matches_with_elo.csv with shape {df_raw.shape}")
print("Columns:", list(df_raw.columns))

# To detect a date column for time-based splitting later
date_col = None
for cand in ["tourney_date", "date", "match_date", "event_date"]:
    if cand in df_raw.columns:
        date_col = cand
        break
if date_col is not None:
    try:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        print(f"Using '{date_col}' as date column for time-based split.")
    except Exception:
        print(f"Could not parse '{date_col}' as datetime; will fall back to random split.")
        date_col = None
else:
    print("No obvious date column found; will use random train/test split.")

# 2. Load feature columns used by the old logistic model
try:
    feature_cols = (
        pd.read_csv("feature_columns.csv", header=None, dtype=str)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
        .dropna()
    )
    feature_cols = feature_cols[(feature_cols != "") & (feature_cols != "0")].tolist()
    print(f"Loaded {len(feature_cols)} base feature columns from feature_columns.csv:")
    print(feature_cols)
except FileNotFoundError:
    raise FileNotFoundError(
        "feature_columns.csv not found. "
        "You should have this from your previous logistic model training."
    )

# Checking all base feature columns exist in the CSV
missing = [c for c in feature_cols if c not in df_raw.columns]
if missing:
    raise ValueError(
        f"The following feature columns are missing in matches_with_elo.csv: {missing}"
    )

# 3. Build a mirrored dataset with labels 
# Positive rows: "Player A = actual winner" (y = 1)
df_pos = df_raw.copy()
df_pos["y"] = 1

# Negative rows: "Player A = actual loser" (y = 0)
df_neg = df_raw.copy()
df_neg["y"] = 0

# For the negative rows, flip the sign of difference-type features
diff_cols = [c for c in ["rank_diff", "elo_diff", "elo_diff_surface"] if c in df_neg.columns]
print("Flipping sign for difference columns in negative class:", diff_cols)

for c in diff_cols:
    df_neg[c] = -df_neg[c]

# Combine mirrored data
df = pd.concat([df_pos, df_neg], ignore_index=True)
print(f"Combined mirrored dataset shape: {df.shape}")
print("Class balance:", df["y"].value_counts().to_dict())

# 4. features (help both winner + total games) 
extra_features = []

if "elo_diff" in df.columns:
    df["elo_diff_abs"] = df["elo_diff"].abs()
    df["elo_diff_sq"] = df["elo_diff"] ** 2
    extra_features += ["elo_diff_abs", "elo_diff_sq"]

if "elo_diff_surface" in df.columns:
    df["elo_diff_surface_abs"] = df["elo_diff_surface"].abs()
    df["elo_diff_surface_sq"] = df["elo_diff_surface"] ** 2
    extra_features += ["elo_diff_surface_abs", "elo_diff_surface_sq"]

if "rank_diff" in df.columns:
    df["rank_diff_abs"] = df["rank_diff"].abs()
    extra_features.append("rank_diff_abs")

# Binary flag for best-of-5 
if "best_of" in df.columns:
    df["is_bo5"] = (df["best_of"] >= 5).astype(int)
    extra_features.append("is_bo5")

# Merge base + extra feature lists for the winner model
feature_cols_gb = feature_cols + extra_features
print(f"Total feature count for GB model (winner): {len(feature_cols_gb)}")
print("Extra engineered features:", extra_features)

# 5. Build X, y for winner model 
X_cls = df[feature_cols_gb]
y_cls = df["y"].astype(int)

# 6. Train/test split for winner model 
if date_col is not None and date_col in df.columns:
    # Time-based split: older matches for train, newest ~20% for test
    df_sorted = df.sort_values(date_col)
    X_sorted = df_sorted[feature_cols_gb]
    y_sorted = df_sorted["y"].astype(int)

    split_idx = int(len(df_sorted) * 0.8)
    X_train_cls, X_test_cls = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train_cls, y_test_cls = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

    print(
        f"Time-based split using '{date_col}' for winner model: "
        f"Train shape {X_train_cls.shape}, Test shape {X_test_cls.shape}"
    )
else:
    # Fallback: standard stratified random split
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    print(f"Random stratified split (winner): Train {X_train_cls.shape}, Test {X_test_cls.shape}")

# 7. Define Gradient Boosting model (winner) 
model_cls = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_iter=500,
    min_samples_leaf=20,
    validation_fraction=0.15,
    early_stopping=True,
    n_iter_no_change=20
)

# 8. Train winner model 
print("Training Gradient Boosting WINNER model...")
model_cls.fit(X_train_cls, y_train_cls)

# 9. Evaluate winner model 
y_pred_cls = model_cls.predict(X_test_cls)
try:
    y_proba_cls = model_cls.predict_proba(X_test_cls)[:, 1]
except AttributeError:
    raw = model_cls.decision_function(X_test_cls)
    y_proba_cls = 1 / (1 + np.exp(-raw))

acc = accuracy_score(y_test_cls, y_pred_cls)
auc = roc_auc_score(y_test_cls, y_proba_cls)

print(f"WINNER model - Accuracy (test): {acc:.3f}")
print(f"WINNER model - ROC AUC (test): {auc:.3f}")

# 10. Save winner model + feature list 
joblib.dump(model_cls, "model_gb.joblib")
pd.Series(feature_cols_gb).to_csv("feature_columns_gb.csv", index=False, header=False)
print("Saved model_gb.joblib and feature_columns_gb.csv")

# 11. Train TOTAL GAMES model (regression)
if "total_games" not in df.columns:
    print("Column 'total_games' not found in matches_with_elo.csv; skipping total-games model.")
else:
    print("\n--- Training TOTAL GAMES model (regression) ---")

    # Features for games: same engineered set (elo closeness + is_bo5 etc.)
    feature_cols_games = feature_cols_gb.copy()
    print(f"Total feature count for GAMES model: {len(feature_cols_games)}")

    X_reg = df[feature_cols_games]
    y_reg = df["total_games"].astype(float)

    if date_col is not None and date_col in df.columns:
        df_sorted_g = df.sort_values(date_col)
        X_sorted_g = df_sorted_g[feature_cols_games]
        y_sorted_g = df_sorted_g["total_games"].astype(float)

        split_idx_g = int(len(df_sorted_g) * 0.8)
        X_train_reg, X_test_reg = X_sorted_g.iloc[:split_idx_g], X_sorted_g.iloc[split_idx_g:]
        y_train_reg, y_test_reg = y_sorted_g.iloc[:split_idx_g], y_sorted_g.iloc[split_idx_g:]

        print(
            f"Time-based split using '{date_col}' for GAMES model: "
            f"Train shape {X_train_reg.shape}, Test shape {X_test_reg.shape}"
        )
    else:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        print(f"Random split (games): Train {X_train_reg.shape}, Test {X_test_reg.shape}")

    games_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        min_samples_leaf=20,
        validation_fraction=0.15,
        early_stopping=True,
        n_iter_no_change=20
    )

    print("Training Gradient Boosting TOTAL GAMES model...")
    games_model.fit(X_train_reg, y_train_reg)

    y_pred_games = games_model.predict(X_test_reg)
    mae_games = mean_absolute_error(y_test_reg, y_pred_games)
    print(f"TOTAL GAMES model - MAE (test): {mae_games:.3f}")

    joblib.dump(games_model, "model_games.joblib")
    pd.Series(feature_cols_games).to_csv(
        "feature_columns_games_gb.csv", index=False, header=False
    )
    print("Saved model_games.joblib and feature_columns_games_gb.csv")

