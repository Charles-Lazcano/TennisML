# scripts/train_sets_games.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN = DATA / "combined_slim_for_model.csv"

df = pd.read_csv(IN, parse_dates=["tourney_date"])
df = df.dropna(subset=["winner_rank","loser_rank","surface","best_of","num_sets"])

# Features available pre-match
df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
FEATS = ["rank_diff", "best_of", "surface"]

def make_model(df_sub, target_values, outpath):
    X = df_sub[FEATS].copy()
    y = df_sub["num_sets"].astype(int)
    df_keep = df_sub[y.isin(target_values)]
    X = df_keep[FEATS]
    y = df_keep["num_sets"].astype(int)

    pre = ColumnTransformer(
        [("surf", OneHotEncoder(handle_unknown="ignore"), ["surface"])],
        remainder="passthrough",
    )
    clf = Pipeline([("prep", pre),
                    ("logreg", LogisticRegression(max_iter=400, multi_class="auto"))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"Saved {outpath.name} • classes {sorted(target_values)} • acc={acc:.3f}")
    joblib.dump(clf, outpath)

# Best-of-3 → 2 or 3 sets
bo3 = df[df["best_of"]==3].copy()
make_model(bo3, {2,3}, ROOT / "model_sets_bo3.joblib")

# Best-of-5 → 3, 4, or 5 sets
bo5 = df[df["best_of"]==5].copy()
make_model(bo5, {3,4,5}, ROOT / "model_sets_bo5.joblib")
