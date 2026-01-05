# scripts/train.py
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

if not IN.exists():
    raise FileNotFoundError(f"Missing {IN}. Run scripts/get_data.py first!")

print("🔹 Loading data...")
df = pd.read_csv(IN, parse_dates=["tourney_date"])
df = df.dropna(subset=["winner_rank","loser_rank","surface","best_of"])
df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
df["label"] = (df["rank_diff"] < 0).astype(int)

feat = ["rank_diff", "best_of", "surface"]
X = df[feat]
y = df["label"]

pre = ColumnTransformer(
    transformers=[("surf", OneHotEncoder(handle_unknown="ignore"), ["surface"])],
    remainder="passthrough"
)
clf = Pipeline([
    ("prep", pre),
    ("logreg", LogisticRegression(max_iter=200))
])

print("🔹 Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Winner model accuracy: {acc:.3f}")

# Save model
joblib.dump(clf, ROOT / "model_logreg.joblib")

# Save feature column names
surf_cats = sorted(df["surface"].dropna().unique().tolist())
feature_cols = ["rank_diff", "best_of"] + [f"surf_{s}" for s in surf_cats]
pd.Series(feature_cols).to_csv(ROOT / "feature_columns.csv", index=False, header=False)

# Save player list
players = pd.concat([
    df[["winner_name","winner_rank"]].rename(columns={"winner_name":"name","winner_rank":"rank"}),
    df[["loser_name","loser_rank"]].rename(columns={"loser_name":"name","loser_rank":"rank"})
], ignore_index=True)
players = (players.dropna(subset=["name"])
           .sort_values(["name","rank"])
           .groupby("name", as_index=False)
           .first())
players.to_csv(ROOT / "players.csv", index=False)

print("✅ Exported: model_logreg.joblib, feature_columns.csv, players.csv")
