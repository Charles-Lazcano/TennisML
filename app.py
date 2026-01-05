# app.py — Tennis Match Predictor (Streamlit)

import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import re
from sklearn.exceptions import InconsistentVersionWarning

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X has feature names, but.*")

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# ── Helpers ─────────────────────────────────────────────────────────────────
def safe_index(options: list, value, default: int = 0) -> int:
    try:
        return int(options.index(value))
    except Exception:
        return int(default)


@st.cache_data(show_spinner=False)
def load_players(path: str = "players.csv") -> pd.DataFrame:
    """
    Load players with rank + Elo (global + per-surface) if available.

    Expected columns (flexible, case-insensitive):
      - name
      - rank (optional)
      - elo / elo_global (optional)
      - elo_hard / elo_clay / elo_grass / elo_carpet (optional)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    name_col = cols.get("name") or df.columns[0]
    rank_col = cols.get("rank")

    elo_global_col = cols.get("elo_global") or cols.get("elo") or cols.get("elo_rating")
    elo_hard_col = cols.get("elo_hard")
    elo_clay_col = cols.get("elo_clay")
    elo_grass_col = cols.get("elo_grass")
    elo_carpet_col = cols.get("elo_carpet")

    out = pd.DataFrame()
    out["name"] = df[name_col].astype(str)

    # Global Elo
    if elo_global_col:
        out["elo_global"] = pd.to_numeric(df[elo_global_col], errors="coerce")
    else:
        out["elo_global"] = np.nan

    # Rank
    if rank_col:
        out["rank"] = pd.to_numeric(df[rank_col], errors="coerce").fillna(9999).astype(int)
    else:
        # derive rank from global elo (descending)
        tmp = out.copy()
        # players with NaN go bottom
        tmp["elo_sort"] = tmp["elo_global"].fillna(tmp["elo_global"].min() - 100)
        tmp = tmp.sort_values("elo_sort", ascending=False).reset_index(drop=True)
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        out = tmp.drop(columns=["elo_sort"])

    # Surface Elo (fallback to global)
    for surf, col in [
        ("Hard", elo_hard_col),
        ("Clay", elo_clay_col),
        ("Grass", elo_grass_col),
        ("Carpet", elo_carpet_col),
    ]:
        target = f"elo_{surf.lower()}"
        if col:
            out[target] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[target] = out["elo_global"]

    # Clean duplicates + sort
    out = (
        out.drop_duplicates("name", keep="first")
        .sort_values("rank")
        .reset_index(drop=True)
    )

    # Ensure no missing Elo
    out["elo_global"] = out["elo_global"].fillna(1500.0)
    for surf in ["hard", "clay", "grass", "carpet"]:
        out[f"elo_{surf}"] = out[f"elo_{surf}"].fillna(out["elo_global"])

    return out[["name", "rank", "elo_global", "elo_hard", "elo_clay", "elo_grass", "elo_carpet"]]


@st.cache_data(show_spinner=False)
def load_feature_columns(path: str = "feature_columns_gb.csv") -> list[str]:
    col = (
        pd.read_csv(path, header=None, dtype=str)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
        .dropna()
    )
    col = col[(col != "") & (col != "0")]
    return col.tolist()


@st.cache_data(show_spinner=False)
def load_feature_columns_games(path: str = "feature_columns_games_gb.csv"):
    try:
        col = (
            pd.read_csv(path, header=None, dtype=str)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .dropna()
        )
        col = col[(col != "") & (col != "0")]
        return col.tolist()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_model(path: str = "model_gb.joblib"):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_sets_model_bo3(path: str = "model_sets_bo3.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_sets_model_bo5(path: str = "model_sets_bo5.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_games_model(path: str = "model_games.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None


# ── Load data/models ─────────────────────────────────────────────────────────
try:
    players_df = load_players("players.csv")
    feature_cols = load_feature_columns("feature_columns_gb.csv")
    clf = load_model("model_gb.joblib")
except Exception as e:
    st.error(f"Failed to load required files: {e}")
    st.stop()

sets_bo3 = load_sets_model_bo3()
sets_bo5 = load_sets_model_bo5()
games_bundle = load_games_model()
games_feature_cols = load_feature_columns_games()  # can be None


# ── Fast lookup ───────────────────────────────────────────────────────────────
RANK_BY_NAME = dict(zip(players_df["name"], players_df["rank"]))
ELO_GLOBAL_BY_NAME = dict(zip(players_df["name"], players_df["elo_global"]))
ELO_SURFACE_BY_NAME = {
    "Hard": dict(zip(players_df["name"], players_df["elo_hard"])),
    "Clay": dict(zip(players_df["name"], players_df["elo_clay"])),
    "Grass": dict(zip(players_df["name"], players_df["elo_grass"])),
    "Carpet": dict(zip(players_df["name"], players_df["elo_carpet"])),
}


# ── Label cleanup helpers ─────────────────────────────────────────────────────
def clean_name(name: str) -> str:
    """Remove any existing '#123 ' prefix."""
    return re.sub(r"^#?\d+\s+", "", str(name)).strip()


def make_label(name: str) -> str:
    """Display '#1 Carlos Alcaraz' etc."""
    r = RANK_BY_NAME.get(name, 9999)
    return f"#{r} {clean_name(name)}"


NAME_BY_LABEL = {make_label(n): n for n in players_df["name"]}
LABELS = list(NAME_BY_LABEL.keys())


# ── Feature Engineering ───────────────────────────────────────────────────────
def build_feature_frame(p1: str, p2: str, surface: str, best_of: int) -> pd.DataFrame:
    """
    Build full feature frame for both winner + total-games models.

    Uses:
      - rank_diff
      - best_of (+ is_bo5)
      - surf_Hard / surf_Clay / surf_Grass / surf_Carpet
      - elo_diff (global)
      - elo_diff_surface
      - elo_diff_abs / elo_diff_sq
      - elo_diff_surface_abs / elo_diff_surface_sq
      - rank_diff_abs
    """
    r1 = int(RANK_BY_NAME.get(p1, 9999))
    r2 = int(RANK_BY_NAME.get(p2, 9999))

    elo1_g = float(ELO_GLOBAL_BY_NAME.get(p1, 1500.0))
    elo2_g = float(ELO_GLOBAL_BY_NAME.get(p2, 1500.0))

    elo1_s = float(ELO_SURFACE_BY_NAME.get(surface, {}).get(p1, elo1_g))
    elo2_s = float(ELO_SURFACE_BY_NAME.get(surface, {}).get(p2, elo2_g))

    # Keep orientation consistent with rank_diff: diff = player2 - player1
    rank_diff = r2 - r1
    elo_diff = elo2_g - elo1_g
    elo_diff_surface = elo2_s - elo1_s

    row = {
        "rank_diff": rank_diff,
        "best_of": int(best_of),
        "elo_diff": elo_diff,
        "elo_diff_surface": elo_diff_surface,
        "elo_diff_abs": abs(elo_diff),
        "elo_diff_sq": elo_diff**2,
        "elo_diff_surface_abs": abs(elo_diff_surface),
        "elo_diff_surface_sq": elo_diff_surface**2,
        "rank_diff_abs": abs(rank_diff),
        "is_bo5": 1 if best_of >= 5 else 0,
    }

    # Surface one-hots
    for s in SURFACES:
        row[f"surf_{s}"] = 1 if s == surface else 0

    return pd.DataFrame([row])


def feature_row(p1: str, p2: str, surface: str, best_of: int) -> pd.DataFrame:
    """Build the feature row expected by the WINNER model."""
    base = build_feature_frame(p1, p2, surface, best_of)
    X = base.reindex(columns=feature_cols, fill_value=0)

    # Ensure numeric types
    for c in ("rank_diff", "best_of", "elo_diff", "elo_diff_surface"):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X


def feature_row_games(p1: str, p2: str, surface: str, best_of: int) -> pd.DataFrame:
    """
    Feature row for TOTAL GAMES model.
    Uses its own feature column list if present, otherwise falls back to winner model's columns.
    """
    base = build_feature_frame(p1, p2, surface, best_of)
    cols = games_feature_cols if games_feature_cols is not None else feature_cols
    return base.reindex(columns=cols, fill_value=0)


def predict_proba(p1: str, p2: str, surface: str, best_of: int) -> float:
    X = feature_row(p1, p2, surface, best_of)
    return float(clf.predict_proba(X)[0, 1])


def feature_row_raw(p1: str, p2: str, surface: str, best_of: int) -> pd.DataFrame:
    """Raw minimal feature row for the sets models (trained separately)."""
    r1 = int(RANK_BY_NAME.get(p1, 9999))
    r2 = int(RANK_BY_NAME.get(p2, 9999))
    return pd.DataFrame([{"rank_diff": r2 - r1, "best_of": int(best_of), "surface": surface}])


def predict_sets_probs(p1: str, p2: str, surface: str, best_of: int):
    model = sets_bo3 if best_of == 3 else sets_bo5
    if model is None:
        return None
    Xraw = feature_row_raw(p1, p2, surface, best_of)
    proba = model.predict_proba(Xraw)[0]
    classes = [int(c) for c in model.classes_]
    return dict(zip(classes, proba))


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎾 Tennis Match Predictor")
st.caption("Pick two players, choose the surface and match format, then click Predict.")

if "p1" not in st.session_state:
    st.session_state.p1 = LABELS[0]
if "p2" not in st.session_state:
    st.session_state.p2 = LABELS[min(1, len(LABELS) - 1)]
if "surface" not in st.session_state:
    st.session_state.surface = "Hard"
if "bestof" not in st.session_state:
    st.session_state.bestof = 3

col1, col2 = st.columns(2)
with col1:
    p1_label = st.selectbox("Player 1", LABELS, index=safe_index(LABELS, st.session_state.p1))
with col2:
    p2_label = st.selectbox("Player 2", LABELS, index=safe_index(LABELS, st.session_state.p2))

st.session_state.p1 = p1_label
st.session_state.p2 = p2_label

p1 = NAME_BY_LABEL[p1_label]
p2 = NAME_BY_LABEL[p2_label]

surface = st.selectbox("Surface", SURFACES, index=safe_index(SURFACES, st.session_state.surface))
st.session_state.surface = surface

best_of = st.radio("Match format", [3, 5], horizontal=True, index=0 if st.session_state.bestof == 3 else 1)
st.session_state.bestof = best_of

a1, a2 = st.columns([1, 1])
with a1:
    if st.button("Swap players"):
        st.session_state.p1, st.session_state.p2 = st.session_state.p2, st.session_state.p1
        st.rerun()
with a2:
    do_predict = st.button("Predict", type="primary")

st.markdown("---")


# ── Prediction ────────────────────────────────────────────────────────────────
if do_predict:
    if p1 == p2:
        st.warning("Please choose two different players.")
    else:
        try:
            # Winner prediction
            proba = predict_proba(p1, p2, surface, best_of)
            st.progress(int(round(proba * 100)))
            winner = p1 if proba >= 0.5 else p2
            st.success(
                f"{winner} is favored • Win probability for {p1} vs {p2} on {surface} (best-of-{best_of}): {proba:.1%}"
            )

            # Sets prediction (optional)
            sets_probs = predict_sets_probs(p1, p2, surface, best_of)
            if sets_probs:
                st.subheader("🧮 Sets prediction")
                if best_of == 3:
                    p2s = float(sets_probs.get(2, 0.0))
                    p3s = float(sets_probs.get(3, 0.0))
                    st.write(f"Best-of-3 · {p2s:.1%} → 2 sets • {p3s:.1%} → 3 sets")
                    st.progress(int(round(p2s * 100)), text="2 sets")
                    st.progress(int(round(p3s * 100)), text="3 sets")
                else:
                    p3s = float(sets_probs.get(3, 0.0))
                    p4s = float(sets_probs.get(4, 0.0))
                    p5s = float(sets_probs.get(5, 0.0))
                    st.write(f"Best-of-5 · {p3s:.1%} → 3 sets • {p4s:.1%} → 4 sets • {p5s:.1%} → 5 sets")
                    st.progress(int(round(p3s * 100)), text="3 sets")
                    st.progress(int(round(p4s * 100)), text="4 sets")
                    st.progress(int(round(p5s * 100)), text="5 sets")

            # ── 🎯 Total Games + Over/Under helper (optional) ───────────────────
            if games_bundle:
                # Support both: plain model or {"model": ..., "mae": ...}
                if isinstance(games_bundle, dict):
                    g_model = games_bundle.get("model")
                    mae = float(games_bundle.get("mae", 3.5))
                else:
                    g_model = games_bundle
                    mae = 3.5

                if g_model is not None:
                    X_games = feature_row_games(p1, p2, surface, best_of)
                    g_pred = float(g_model.predict(X_games)[0])

                    lo_raw = g_pred - mae
                    hi = g_pred + mae
                    lo = max(0.0, lo_raw)

                    st.subheader("🎯 Total games prediction")
                    st.write(f"Expected total games: {g_pred:.1f} (± {mae:.1f}, ≈ {lo:.1f}–{hi:.1f})")

                    # clamp to [0, 50] for progress bar
                    g_clamped = min(max(g_pred, 0.0), 50.0)
                    st.progress(int(round(g_clamped / 50 * 100)))
                    st.caption("Progress bar scaled to 50 games (rough upper bound for Bo5).")

                    with st.expander("📊 Over/Under helper"):
                        default_line = int(round(g_clamped))
                        line = st.slider(
                            "Set your O/U line",
                            min_value=0,
                            max_value=50,
                            value=default_line,
                            step=1,
                        )
                        if g_pred > line:
                            st.success(f"Leans **Over {line}** (estimate {g_pred:.1f}).")
                        elif g_pred < line:
                            st.warning(f"Leans **Under {line}** (estimate {g_pred:.1f}).")
                        else:
                            st.info(f"Right on the line ~{line}.")
            else:
                st.info("Total games model not found. (Optional) retrain to create model_games.joblib.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ── Info ──────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Why might I see a scikit-learn version warning?"):
    st.write(
        "You trained the model with one version of scikit-learn and are loading it with another. "
        "That can still work, but for maximum safety, retrain in the same environment you deploy."
    )

st.caption(
    "Winner model: Gradient Boosting (HistGradientBoostingClassifier) • Inputs include rank difference, "
    "Elo differences (global + surface), best-of, and surface one-hots. "
    "Sets model: separate classifier for Bo3/Bo5 (optional). "
    "Total games: Gradient Boosting regressor using the same rich features, with an ±MAE band "
    "and Over/Under helper."
)
