#!/usr/bin/env python3
# build_elo_and_retrain.py
# ------------------------------------------------------------
# Computes Elo ratings (global + per-surface) from your historical matches,
# exports current Elo table for each player, joins Elo to matches, and retrains
# a logistic regression model that uses: rank_diff, best_of, surface one-hots,
# elo_diff (global), and elo_diff_surface (surface-aware).
#
# Inputs:
#   - combined_matches_1968_2025.csv    (in project root OR ./data/)
#
# Outputs (default to project root, or use --outdir):
#   - elo_current.csv
#   - matches_with_elo.csv
#   - feature_columns.csv
#   - model_logreg.joblib
#
# Usage:
#   python build_elo_and_retrain.py
#   python build_elo_and_retrain.py --input data\combined_matches_1968_2025.csv
#
# Optional flags:
#   --K 40                   Elo K-factor (default 40)
#   --seed 42                Random seed
#   --val_year_min 2015      Evaluate on modern era (>= year)
#   --limit_year_min 2000    (Optional) Only process matches from this year onward (speed-up)
#   --outdir data            Write artifacts into this folder instead of project root
# ------------------------------------------------------------

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="combined_matches_1968_2025.csv",
                    help="Path or filename of the historical matches CSV (root or ./data/).")
    ap.add_argument("--K", type=float, default=40.0, help="Elo K-factor.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--val_year_min", type=int, default=2015,
                    help="Min year for validation metrics (modern-era check).")
    ap.add_argument("--limit_year_min", type=int, default=None,
                    help="If set, only process matches with year >= this (speed-up).")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Directory to write outputs (default: current directory).")
    return ap.parse_args()

def first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("Could not find input CSV in any of: " + ", ".join(map(str, candidates)))

def normalize_surface(s):
    if pd.isna(s):
        return None
    s = str(s).strip().title()
    aliases = {
        "Hardcourt": "Hard", "Hard Court": "Hard",
        "Clay Court": "Clay", "Grass Court": "Grass",
        "Carpet Court": "Carpet"
    }
    s = aliases.get(s, s)
    return s if s in SURFACES else None

def compute_elo(df: pd.DataFrame, K: float = 40.0):
    """Compute global + per-surface Elo in chronological order; return (elo_snap, elo_current)."""
    elo_global = defaultdict(lambda: 1500.0)
    elo_surface = {s: defaultdict(lambda: 1500.0) for s in SURFACES}

    rows = []
    for _, row in df.iterrows():
        w = row["winner_name"]; l = row["loser_name"]; s = row["surface"]
        if pd.isna(w) or pd.isna(l) or pd.isna(s):
            continue

        Rw_g = elo_global[w]; Rl_g = elo_global[l]
        Rw_s = elo_surface[s][w]; Rl_s = elo_surface[s][l]

        Ew_g = 1.0 / (1.0 + 10 ** ((Rl_g - Rw_g) / 400.0))
        Ew_s = 1.0 / (1.0 + 10 ** ((Rl_s - Rw_s) / 400.0))

        # winner gets 1, loser gets 0
        elo_global[w] = Rw_g + K * (1.0 - Ew_g)
        elo_global[l] = Rl_g - K * (1.0 - Ew_g)

        elo_surface[s][w] = Rw_s + K * (1.0 - Ew_s)
        elo_surface[s][l] = Rl_s - K * (1.0 - Ew_s)

        rows.append({
            "tourney_date": row["tourney_date"],
            "surface": s,
            "winner_name": w, "loser_name": l,
            "winner_elo": elo_global[w], "loser_elo": elo_global[l],
            "winner_elo_surface": elo_surface[s][w],
            "loser_elo_surface": elo_surface[s][l],
        })

    elo_snap = pd.DataFrame(rows)

    # Latest rating per player (global + per-surface)
    players = set(df["winner_name"].dropna()) | set(df["loser_name"].dropna())
    records = []
    for p in players:
        rec = {"player": p, "elo_global": elo_global[p]}
        for s in SURFACES:
            rec[f"elo_{s.lower()}"] = elo_surface[s][p]
        records.append(rec)
    elo_current = pd.DataFrame(records)
    return elo_snap, elo_current

def main():
    args = parse_args()
    np.random.seed(args.seed)

    root = Path.cwd()
    data_dir = root / "data"
    # Try provided path, then ./data/<name>, then ./data/combined_matches_1968_2025.csv
    input_path = first_existing(
        Path(args.input),
        data_dir / args.input,
        data_dir / "combined_matches_1968_2025.csv"
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Loading matches from: {input_path}")
    df = pd.read_csv(input_path)

    # Basic cleaning / normalization
    df["surface"] = df["surface"].apply(normalize_surface)
    df = df[df["surface"].isin(SURFACES)].copy()

    # chronological
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df["year"] = df["tourney_date"].dt.year

    # Optional speed-up: limit to recent years
    if args.limit_year_min is not None:
        before = len(df)
        df = df[df["year"] >= int(args.limit_year_min)].reset_index(drop=True)
        print(f"[i] Year filter applied: >= {args.limit_year_min} ({before} -> {len(df)} matches)")

    # Ensure rank / best_of present
    for col in ["winner_rank", "loser_rank", "best_of"]:
        if col not in df.columns:
            df[col] = np.nan
    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce").fillna(1000).astype(int)
    df["loser_rank"]  = pd.to_numeric(df["loser_rank"],  errors="coerce").fillna(1000).astype(int)
    df["best_of"]     = pd.to_numeric(df["best_of"],     errors="coerce").fillna(3).astype(int)

    # Compute Elo
    print("[i] Computing Elo ratings (global + per-surface)...")
    elo_snap, elo_current = compute_elo(df, K=args.K)

    # Align and merge Elo to matches
    df = df.reset_index(drop=True)
    elo_snap = elo_snap.reset_index(drop=True)
    merged = df.join(elo_snap[["winner_elo","loser_elo","winner_elo_surface","loser_elo_surface"]])

    merged["elo_diff"] = merged["winner_elo"] - merged["loser_elo"]
    merged["elo_diff_surface"] = merged["winner_elo_surface"] - merged["loser_elo_surface"]

    # Surface one-hots
    for s in SURFACES:
        merged[f"surf_{s}"] = (merged["surface"] == s).astype(int)

    # Rank diff
    merged["rank_diff"] = merged["winner_rank"] - merged["loser_rank"]

    # Save artifacts
    elo_current_path = outdir / "elo_current.csv"
    matches_with_elo_path = outdir / "matches_with_elo.csv"
    elo_current.to_csv(elo_current_path, index=False)
    merged.to_csv(matches_with_elo_path, index=False)
    print(f"[✓] Saved: {elo_current_path}")
    print(f"[✓] Saved: {matches_with_elo_path}")

    # Build balanced training matrix (winner vs swapped loser perspective)
    a = pd.DataFrame({
        "rank_diff": merged["rank_diff"],
        "elo_diff": merged["elo_diff"],
        "elo_diff_surface": merged["elo_diff_surface"],
        "best_of": merged["best_of"],
        "surf_Hard": merged["surf_Hard"],
        "surf_Clay": merged["surf_Clay"],
        "surf_Grass": merged["surf_Grass"],
        "surf_Carpet": merged["surf_Carpet"],
        "y": 1
    })
    b = pd.DataFrame({
        "rank_diff": -merged["rank_diff"],
        "elo_diff": -merged["elo_diff"],
        "elo_diff_surface": -merged["elo_diff_surface"],
        "best_of": merged["best_of"],
        "surf_Hard": merged["surf_Hard"],
        "surf_Clay": merged["surf_Clay"],
        "surf_Grass": merged["surf_Grass"],
        "surf_Carpet": merged["surf_Carpet"],
        "y": 0
    })
    data = pd.concat([a, b], ignore_index=True).dropna()

    # Feature list (order matters)
    feature_columns = [
        "rank_diff",
        "elo_diff",
        "elo_diff_surface",
        "best_of",
        "surf_Hard", "surf_Clay", "surf_Grass", "surf_Carpet",
    ]
    feature_columns_path = outdir / "feature_columns.csv"
    pd.Series(feature_columns).to_csv(feature_columns_path, index=False, header=False)
    print(f"[✓] Saved: {feature_columns_path}")

    X = data[feature_columns].astype(float).values
    y = data["y"].astype(int).values

    # Train logistic regression
    print("[i] Training logistic regression with Elo features...")
    # Use lbfgs; n_jobs only applies to certain solvers, so we omit it for compatibility.
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)

    model_path = outdir / "model_logreg.joblib"
    joblib.dump(model, model_path)
    print(f"[✓] Saved: {model_path}")

    # Modern-era quick eval (optional)
    modern_mask = merged["year"] >= int(args.val_year_min)
    if modern_mask.any():
        a_m = a[modern_mask]
        b_m = b[modern_mask]
        data_m = pd.concat([a_m, b_m], ignore_index=True).dropna()
        Xm = data_m[feature_columns].astype(float).values
        ym = data_m["y"].astype(int).values
        proba = model.predict_proba(Xm)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(ym, pred)
        auc = roc_auc_score(ym, proba)
        ll  = log_loss(ym, proba)
        print(f"[✓] Validation (year >= {args.val_year_min}): acc={acc:.4f}  auc={auc:.4f}  logloss={ll:.4f}  n={len(ym)}")
    else:
        print("[i] No matches found for validation range; skipping modern-era eval.")

    print("[✓] Done.")

if __name__ == "__main__":
    main()
