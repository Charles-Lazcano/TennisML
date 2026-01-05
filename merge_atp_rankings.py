#!/usr/bin/env python3
# Merge ATP rankings into your players.csv / elo_current.csv

import argparse
import pandas as pd
import re
from pathlib import Path

def normalize_name(s: str) -> str:
    s = str(s or "")
    s = s.lower().strip()
    # remove punctuation & multiple spaces; keep letters/spaces
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def load_players(players_path: str, elo_path: str):
    # players.csv is the file your app uses for dropdown; if missing, build from Elo
    if Path(players_path).exists():
        p = pd.read_csv(players_path)
        # ensure we have a raw 'player' column; if not, derive from 'name'
        if "player" not in p.columns and "name" in p.columns:
            p["player"] = p["name"].str.replace(r"^#\d+\s+", "", regex=True)
    else:
        # fallback: build minimal players table from Elo
        elo = pd.read_csv(elo_path)
        elo = elo.sort_values("elo_global", ascending=False).reset_index(drop=True)
        elo["rank"] = elo.index + 1
        p = pd.DataFrame({
            "name": elo.apply(lambda r: f"#{r['rank']} {r['player']}", axis=1),
            "player": elo["player"],
            "elo_global": elo["elo_global"],
            "elo_hard": elo.get("elo_hard"),
            "elo_clay": elo.get("elo_clay"),
            "elo_grass": elo.get("elo_grass"),
            "elo_carpet": elo.get("elo_carpet"),
        })
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="players.csv", help="Existing players list (output of your sort script)")
    ap.add_argument("--elo", default="elo_current.csv", help="Elo table (fallback if players.csv missing)")
    ap.add_argument("--ranks", required=True, help="CSV with columns: player,atp_rank,atp_points,asof_date")
    ap.add_argument("--output", default="players.csv", help="Where to write merged players file")
    args = ap.parse_args()

    players = load_players(args.players, args.elo).copy()
    ranks = pd.read_csv(args.ranks)

    # normalize names for join (robust against accents/punctuation)
    players["_key"] = players["player"].map(normalize_name)
    ranks["_key"] = ranks["player"].map(normalize_name)

    merged = players.merge(
        ranks[["_key", "atp_rank", "atp_points", "asof_date"]],
        on="_key", how="left"
    ).drop(columns=["_key"])

    # Rebuild display 'name' column to include ATP rank if available
    # (Keeps your #<Elo rank> prefix but shows ATP in parentheses)
    def display(row):
        base = str(row.get("name", row.get("player", "")))
        if pd.notna(row.get("atp_rank")):
            return f"{base}  (ATP #{int(row['atp_rank'])})"
        return base

    merged["name"] = merged.apply(display, axis=1)

    merged.to_csv(args.output, index=False)
    print(f"[OK] Wrote merged file: {args.output} (ATP ranks attached where matched)")

if __name__ == "__main__":
    main()
