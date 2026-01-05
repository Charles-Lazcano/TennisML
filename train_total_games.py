# scripts/train_total_games.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN = DATA / "combined_slim_for_model.csv"

df = pd.read_csv(IN, parse_dates=["tourney_date"])
df = df.dropna(subset=["winner_rank","loser_rank","surface","best_of","total_games"])

# Features available pre-match
df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
FEATS = ["rank_diff", "best_of", "surface"]

X = df[FEATS].copy()
y = df["total_games"].astype(float)

pre = ColumnTransformer(
    [("surf", OneHotEncoder(handle_unknown="ignore"), ["surface"])],
    remainder="passthrough",
)

reg = Pipeline([
    ("prep", pre),
    ("gbr", GradientBoostingRegressor(random_state=42))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
reg.fit(Xtr, ytr)
pred = reg.predict(Xte)
mae = mean_absolute_error(yte, pred)
print(f"✅ Total games MAE: {mae:.2f}")

# Save model and a simple MAE for UI range
joblib.dump({"model": reg, "mae": float(mae)}, ROOT / "model_games.joblib")
print("Saved model_games.joblib")
