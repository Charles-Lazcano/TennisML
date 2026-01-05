#!/usr/bin/env python3
"""
Sort players by Elo and output a players.csv your Streamlit app can use.

New:
- --active-only            : keep only players with >= --min-recent matches in the last --recent-years (defaults: 5 in 3y)

Other features:
- Sort by elo_global (descending)
- Optional filters:
    --matches <csv>        : path to matches file to compute match counts
    --min-matches <int>    : minimum total matches to include (default: 0)
    --recent-years <int>   : only count matches within last N calendar years for 'matches_recent' (default: 3)
    --min-recent <int>     : minimum matches within last N years (default: 5)
- Output:
    players.csv with columns:
      name (display "#<rank> <player>"), player, rank, elo_global, elo_hard, elo_clay, elo_grass, elo_carpet,
      matches_total, matches_recent

Examples:
    python sort_players_by_elo.py --elo elo_current.csv --active-only --matches combined_matches_1968_2025.csv
    python sort_players_by_elo.py --active-only  # uses defaults: 3y window, >=5 matches
"""

import argparse
import datetime as dt
import sys
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Sort players by Elo and export players.csv")
    p.add_argument("--elo", default="elo_current.csv", help="Path to Elo table CSV (default: elo_current.csv)")
    p.add_argument("--matches", default=None, help="Path to matches CSV (optional, for match counts)")
    p.add_argument("--min-matches", type=int, default=0, help="Min total matches to include (default: 0)")
    p.add_argument("--recent-years", type=int, default=3, help="Years for recent window (default: 3)")
    p.add_argument("--min-recent", type=int, default=5, help="Min recent matches to include (default: 5)")
    p.add_argument("--active-only", action="store_true", help="Keep only players active in recent window")
    p.add_argument("--output", default="players.csv", help="Output CSV path (default: players.csv)")
    return p.parse_args()


def read_elo(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Required columns
    required = ["player", "elo_global"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Elo CSV '{path}' missing required columns: {missing}. Found: {list(df.columns)}")

    # Ensure optional per-surface columns exist
    for c in ["elo_hard", "elo_clay", "elo_grass", "elo_carpet"]:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def parse_tourney_date(series: pd.Series) -> pd.Series:
    """Accept formats like 20240915, '2024-09-15', '20240915'."""
    s = series.copy()
    if pd.api.types.is_integer_dtype(s):
        s = s.astype(str)
    dt1 = pd.to_datetime(s, errors="coerce", format="%Y%m%d")
    mask = dt1.isna()
    dt2 = pd.to_datetime(s[mask], errors="coerce", format="%Y-%m-%d")
    dt1.loc[mask] = dt2
    dt1 = pd.to_datetime(dt1, errors="coerce")  # final fallback
    return dt1


def detect_name_cols(m: pd.DataFrame):
    candidates = [
        ("winner_name", "loser_name"),
        ("w_name", "l_name"),
        ("winner", "loser"),
        ("p1_name", "p2_name"),
        ("player1", "player2"),
        ("p1", "p2"),
    ]
    for w, l in candidates:
        if w in m.columns and l in m.columns:
            return w, l
    raise ValueError(
        f"Could not find winner/loser name columns. Tried common variants. Found: {list(m.columns)}"
    )


def detect_date_col(m: pd.DataFrame):
    for c in ["tourney_date", "date", "match_date", "start_date"]:
        if c in m.columns:
            return c
    return None


def compute_counts(matches_path: str, recent_years: int) -> pd.DataFrame:
    """
    Returns per-player:
      matches_total, matches_recent (within `recent_years`), last_match_date
    """
    m = pd.read_csv(matches_path, low_memory=False)
    wcol, lcol = detect_name_cols(m)

    # total counts
    winners = m[[wcol]].rename(columns={wcol: "player"})
    losers  = m[[lcol]].rename(columns={lcol: "player"})
    all_players = pd.concat([winners["player"], losers["player"]], ignore_index=True)
    total_counts = all_players.value_counts(dropna=False).rename_axis("player").reset_index(name="matches_total")

    # date parsing for recent counts
    tdate_col = detect_date_col(m)
    if tdate_col is None:
        recent_counts = pd.DataFrame(columns=["player", "matches_recent"])
        last_dates = pd.DataFrame(columns=["player", "last_match_date"])
    else:
        m["_tdate"] = parse_tourney_date(m[tdate_col])
        cutoff = dt.datetime.now() - dt.timedelta(days=365 * int(max(recent_years, 0)))
        recent = m.loc[m["_tdate"] >= cutoff]

        w_recent = recent[[wcol]].rename(columns={wcol: "player"})
        l_recent = recent[[lcol]].rename(columns={lcol: "player"})
        recent_counts = (
            pd.concat([w_recent["player"], l_recent["player"]], ignore_index=True)
            .value_counts()
            .rename_axis("player")
            .reset_index(name="matches_recent")
        )

        last_dates = (
            pd.concat([
                m[[wcol, "_tdate"]].rename(columns={wcol: "player"}),
                m[[lcol, "_tdate"]].rename(columns={lcol: "player"})
            ], ignore_index=True)
            .groupby("player")["_tdate"].max()
            .rename("last_match_date")
            .reset_index()
        )

    out = total_counts.merge(recent_counts, on="player", how="left") \
                      .merge(last_dates, on="player", how="left")
    out["matches_recent"] = out["matches_recent"].fillna(0).astype(int)
    out["matches_total"]  = out["matches_total"].fillna(0).astype(int)
    return out


def main():
    args = parse_args()

    # 1) Load Elo
    try:
        elo = read_elo(args.elo)
    except Exception as e:
        print(f"[ERROR] Reading Elo: {e}", file=sys.stderr)
        sys.exit(1)

    df = elo.copy()

    # 2) Attach match counts (if provided)
    if args.matches:
        try:
            counts = compute_counts(args.matches, args.recent_years)
            df = df.merge(counts, on="player", how="left")
        except Exception as e:
            print(f"[WARN] Could not compute match counts from '{args.matches}': {e}")
            df["matches_total"] = 0
            df["matches_recent"] = 0
            df["last_match_date"] = pd.NaT
    else:
        # If no matches file, create empty counters so filtering is still consistent
        df["matches_total"] = 0
        df["matches_recent"] = 0
        df["last_match_date"] = pd.NaT

    # 3) Global filters
    if args.min_matches > 0:
        df = df.loc[df["matches_total"] >= int(args.min_matches)]

    # Active-only = enforce recent window threshold
    if args.active_only:
        # Use provided thresholds; if no matches data, this naturally filters out none (all 0)
        df = df.loc[df["matches_recent"] >= int(args.min_recent)]

    # 4) Rank by global Elo
    df = df.sort_values("elo_global", ascending=False).reset_index(drop=True)
    df["rank"] = (df.index + 1).astype(int)

    # 5) Output for your app
    df_out = pd.DataFrame({
        "name": df.apply(lambda r: f"#{r['rank']} {r['player']}", axis=1),
        "player": df["player"],
        "rank": df["rank"],
        "elo_global": df["elo_global"],
        "elo_hard": df["elo_hard"],
        "elo_clay": df["elo_clay"],
        "elo_grass": df["elo_grass"],
        "elo_carpet": df["elo_carpet"],
        "matches_total": df["matches_total"],
        "matches_recent": df["matches_recent"],
        "last_match_date": df["last_match_date"],
    })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote sorted players to: {out_path} ({len(df_out):,} rows)")
    if args.active_only:
        print(f"[INFO] Active-only: >= {args.min_recent} matches in the last {args.recent_years} year(s).")


if __name__ == "__main__":
    main()
