#!/usr/bin/env python

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# ============ #
# 0) CONFIG    #
# ============ #
DATA_LOCAL = os.path.join("data", "STATS_NHL_ALL_19_25.xlsx")
DATA_FALLBACK_URL = (
    "https://raw.githubusercontent.com/CedrikCampeau31/STATS_SPORTS/main/STATS_NHL_ALL_19_25.xlsx"
)
SHEET = 0
SKIPROWS = 1

BASE_MODELS = os.path.join("artifacts_models")
os.makedirs(BASE_MODELS, exist_ok=True)


# ============ #
# 1) LOAD      #
# ============ #
def load_df():
    path = DATA_LOCAL if os.path.exists(DATA_LOCAL) else DATA_FALLBACK_URL
    print(f"[LOAD] {path}")
    df_raw = pd.read_excel(path, sheet_name=SHEET, skiprows=SKIPROWS, engine="openpyxl")
    return df_raw.copy()


# ========================================= #
# 2) CLEAN & FEATURE ENGINEERING (condensé) #
# ========================================= #
RENAME = {
    "Rk": "rank",
    "Name": "playerName",
    "Team": "team",
    "Age": "age",
    "Pos": "pos",
    "GP": "gp",
    "G": "g",
    "A": "a",
    "P": "pts",
    "PIM": "pim",
    "+/-": "pm",
    "TOI": "toi",
    "ES": "es_pts",
    "PP": "pp_pts",
    "SH": "sh_pts",
    "ESG": "esg",
    "PPG": "ppg",
    "SHG": "shg",
    "GWG": "gwg",
    "OTG": "otg",
    "ESA": "esa",
    "PPA": "ppa",
    "SHA": "sha",
    "GWA": "gwa",
    "OTA": "ota",
    "ESP": "esp",
    "PPP": "ppp",
    "SHP": "shp",
    "GWP": "gwp",
    "OTP": "otp",
    "PPP%": "ppp_pct",
    "G/60": "g_per60",
    "A/60": "a_per60",
    "P/60": "p_per60",
    "ESG/60": "esg_per60",
    "ESA/60": "esa_per60",
    "ESP/60": "esp_per60",
    "PPG/60": "ppg_per60",
    "PPA/60": "ppa_per60",
    "PPP/60": "ppp_per60",
    "G/GP": "g_pergp",
    "A/GP": "a_pergp",
    "P/GP": "p_pergp",
    "SHOTS": "shots",
    "SH%": "sh_pct",
    "HITS": "hits",
    "BS": "blocks",
    "FOW": "fow",
    "FOL": "fol",
    "FO%": "fo_pct",
    "Season": "seasonId",
}

KEEP = [
    "rank",
    "playerName",
    "team",
    "age",
    "pos",
    "seasonId",
    "gp",
    "g",
    "a",
    "pts",
    "pim",
    "pm",
    "toi",
    "es_pts",
    "pp_pts",
    "sh_pts",
    "esg",
    "ppg",
    "shg",
    "gwg",
    "otg",
    "esa",
    "ppa",
    "sha",
    "gwa",
    "ota",
    "esp",
    "ppp",
    "shp",
    "gwp",
    "otp",
    "ppp_pct",
    "g_per60",
    "a_per60",
    "p_per60",
    "esg_per60",
    "esa_per60",
    "esp_per60",
    "ppg_per60",
    "ppa_per60",
    "ppp_per60",
    "g_pergp",
    "a_pergp",
    "p_pergp",
    "shots",
    "sh_pct",
    "hits",
    "blocks",
    "fow",
    "fol",
    "fo_pct",
]


def toi_to_minutes(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", s):
        h, m, sec = s.split(":")
        return int(h) * 60 + int(m) + int(sec) / 60.0
    if re.match(r"^\d{1,3}:\d{2}$", s):
        m, sec = s.split(":")
        return int(m) + int(sec) / 60.0
    return np.nan


def _find_col(df, candidates):
    norm = {re.sub(r"[\s_]+", "", str(c)).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_]+", "", cand).lower()
        if key in norm:
            return norm[key]
    for cand in candidates:
        key = re.sub(r"[\s_]+", "", cand).lower()
        for k, orig in norm.items():
            if key in k:
                return orig
    return None


def ensure_season_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    season_col = _find_col(df, ["seasonId", "season_id", "season", "Season"])
    if season_col is None:
        raise ValueError("Pas de colonne saison trouvée.")
    if season_col != "seasonId":
        df = df.rename(columns={season_col: "seasonId"})

    if not pd.api.types.is_numeric_dtype(df["seasonId"]):
        s = df["seasonId"].astype(str)
        if s.str.fullmatch(r"\d+").all():
            df["seasonId"] = pd.to_numeric(s, errors="coerce")

    player_col = _find_col(
        df, ["playerId", "player_id", "skaterId", "nhlId", "playerName", "Name", "name", "player"]
    )
    if player_col is None:
        raise ValueError("Pas de colonne identifiant joueur (playerName/Name/...).")

    df = df.sort_values([player_col, "seasonId"])
    if "seasonId_prev" not in df.columns:
        df["seasonId_prev"] = df.groupby(player_col)["seasonId"].shift(1)
    return df, player_col


def prepare_df(df_raw):
    df = df_raw.rename(columns={k: v for k, v in RENAME.items() if k in df_raw.columns})
    df = df[[c for c in KEEP if c in df.columns]].copy()

    if "seasonId" in df.columns:
        if pd.api.types.is_numeric_dtype(df["seasonId"]):
            df["seasonId"] = df["seasonId"].fillna(0).astype(int)

    for pct in ["sh_pct", "fo_pct", "ppp_pct"]:
        if pct in df.columns:
            df[pct] = pd.to_numeric(
                df[pct].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )

    num_cols = [c for c in df.columns if c not in ["playerName", "team", "pos", "toi", "seasonId"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "toi" in df.columns:
        df["toi_min_total"] = df["toi"].apply(toi_to_minutes)

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip().str.upper()
    if "pos" in df.columns:
        df["pos"] = df["pos"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["playerName", "team"])
    df = df[df["gp"].fillna(0) >= 0]

    df, player_col = ensure_season_cols(df)

    stat_cols = [
        c
        for c in [
            "gp",
            "g",
            "a",
            "pts",
            "shots",
            "hits",
            "blocks",
            "fo_pct",
            "pm",
            "pim",
            "g_per60",
            "a_per60",
            "p_per60",
            "esp",
            "ppp",
            "shp",
            "toi_min_total",
        ]
        if c in df.columns
    ]
    for col in stat_cols:
        df[f"{col}_prev"] = df.groupby(player_col)[col].shift(1)
        df[f"d_{col}"] = df[col] - df[f"{col}_prev"]

    df["team_id"] = df["team"].astype("category").cat.codes
    return df


# =========================== #
# 3) ML t-1 → t (Bloc 6)      #
# =========================== #
import warnings

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES_BASE = ["p_pergp", "esp", "a", "p_per60", "ppp", "a_per60", "gwp", "esa", "g", "shots"]
RANDOM_STATE = 42
args = None  # rempli par argparse


def macro_position(pos_str):
    if pd.isna(pos_str):
        return "UNK"
    p = str(pos_str).strip().upper()
    if p in {"D", "LD", "RD", "DEF"}:
        return "DEF"
    if p in {"C", "LW", "RW", "W", "F", "FWD"}:
        return "FWD"
    return "DEF" if p.startswith("D") else "FWD"


def choose_k(n_rows):
    if n_rows >= 180:
        return 3
    if n_rows >= 60:
        return 2
    return 1


def build_tminus1_to_t(df):
    base_cols = [
        c
        for c in [
            "playerName",
            "team",
            "team_id",
            "seasonId",
            "age",
            "pos",
            "gp",
            "g",
            "a",
            "pts",
            "pim",
            "pm",
            "es_pts",
            "pp_pts",
            "sh_pts",
            "esg",
            "ppg",
            "shg",
            "gwg",
            "otg",
            "esa",
            "ppa",
            "sha",
            "gwa",
            "ota",
            "esp",
            "ppp",
            "shp",
            "gwp",
            "otp",
            "ppp_pct",
            "g_per60",
            "a_per60",
            "p_per60",
            "esg_per60",
            "esa_per60",
            "esp_per60",
            "ppg_per60",
            "ppa_per60",
            "ppp_per60",
            "g_pergp",
            "a_pergp",
            "p_pergp",
            "shots",
            "sh_pct",
            "hits",
            "blocks",
            "fow",
            "fol",
            "fo_pct",
            "toi_min_total",
        ]
        if c in df.columns
    ]
    d0 = df[base_cols].copy()
    feat_cols = [c for c in d0.columns if c not in ["playerName", "seasonId"]]
    d_prev = d0.rename(columns={c: f"{c}_prev" for c in feat_cols})
    d_prev["seasonId"] = d0["seasonId"] + 101
    label = d0[["playerName", "seasonId", "pts"]].rename(columns={"pts": "pts_target"})
    return label.merge(d_prev, on=["playerName", "seasonId"], how="inner")


def season_splits(df_t):
    seasons = sorted([int(s) for s in df_t["seasonId"].dropna().unique()])
    if len(seasons) < 3:
        raise ValueError(f"Pas assez de saisons pour split (trouvées: {seasons})")
    return seasons[:-2], seasons[-2], seasons[-1]


def metrics(y_true, y_pred):
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_training(df):
    df_t = build_tminus1_to_t(df).copy()

    feat_prev_all = [f"{c}_prev" for c in FEATURES_BASE if f"{c}_prev" in df_t.columns]

    # engineered features
    if {"pts_prev", "toi_min_total_prev"}.issubset(df_t.columns):
        df_t["pts_prev_per60"] = 60.0 * df_t["pts_prev"] / df_t["toi_min_total_prev"].replace(
            0, np.nan
        )
        df_t["pts_prev_per60"] = df_t["pts_prev_per60"].fillna(0)
        if "pts_prev_per60" not in feat_prev_all:
            feat_prev_all.append("pts_prev_per60")
    if "age_prev" in df_t.columns:
        df_t["age_prev2"] = df_t["age_prev"] ** 2
        if "age_prev2" not in feat_prev_all:
            feat_prev_all.append("age_prev2")

    if not feat_prev_all:
        raise ValueError("Aucune feature *_prev disponible.")

    for c in feat_prev_all + ["pts_target", "pts_prev", "age_prev", "toi_min_total_prev"]:
        if c in df_t.columns:
            df_t[c] = pd.to_numeric(df_t[c], errors="coerce")

    df_t["macro_pos"] = df_t.get("pos_prev", df_t.get("pos", np.nan)).apply(macro_position)

    if "pts_prev" not in df_t.columns:
        if {"g_prev", "a_prev"}.issubset(df_t.columns):
            df_t["pts_prev"] = pd.to_numeric(df_t["g_prev"], errors="coerce") + pd.to_numeric(
                df_t["a_prev"], errors="coerce"
            )
        else:
            raise ValueError("Il manque 'pts_prev' et (g_prev,a_prev) pour le reconstruire.")

    train_seasons, val_season, test_season = season_splits(df_t)
    m_train = df_t["seasonId"].isin(train_seasons)
    m_val = df_t["seasonId"].eq(val_season)
    m_test = df_t["seasonId"].eq(test_season)

    # Clustering tiers sur TRAIN
    tier_models, tier_centers = {}, {}
    df_t["tier"] = "All"
    for mp in ["FWD", "DEF"]:
        pts_train = df_t.loc[m_train & (df_t["macro_pos"] == mp), "pts_prev"].dropna().values
        k = choose_k(len(pts_train))
        if k == 1:
            tier_models[mp] = None
            tier_centers[mp] = np.array([np.nan])
            continue
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=(5 if args and args.fast else 10),
        ).fit(pts_train.reshape(-1, 1))
        tier_models[mp] = km
        tier_centers[mp] = km.cluster_centers_.flatten()
        m_mp = df_t["macro_pos"] == mp
        labs = km.predict(df_t.loc[m_mp, "pts_prev"].fillna(0).values.reshape(-1, 1))
        order = np.argsort(km.cluster_centers_.flatten())
        mapping = {
            cl: name for cl, name in zip(order, ["Low", "Mid", "High"][: len(order)], strict=False)
        }
        df_t.loc[m_mp, "tier"] = [mapping[i] for i in labs]

    # Entraînement hiérarchique
    models = {}
    X_tr = df_t.loc[m_train, feat_prev_all].astype(float).fillna(0)
    y_tr = df_t.loc[m_train, "pts_target"].astype(float)
    models[("GLOBAL", "_")] = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_iter=(args.max_iter if args else 100),
    ).fit(X_tr, y_tr)

    for mp in ["FWD", "DEF"]:
        idx = m_train & (df_t["macro_pos"] == mp)
        if idx.sum() >= 100:
            models[(mp, "_")] = HistGradientBoostingRegressor(
                random_state=RANDOM_STATE,
                max_iter=(args.max_iter if args else 100),
            ).fit(
                df_t.loc[idx, feat_prev_all].astype(float).fillna(0),
                df_t.loc[idx, "pts_target"].astype(float),
            )
    for mp in ["FWD", "DEF"]:
        tiers = ["Low", "Mid", "High"] if tier_models.get(mp) is not None else ["All"]
        for tname in tiers:
            idx = m_train & (df_t["macro_pos"] == mp) & (df_t["tier"] == tname)
            if idx.sum() >= 50:
                models[(mp, tname)] = HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=(args.max_iter if args else 100),
                ).fit(
                    df_t.loc[idx, feat_prev_all].astype(float).fillna(0),
                    df_t.loc[idx, "pts_target"].astype(float),
                )

    # base: meilleur modèle hiérarchique disponible
    def _pred_with_key(row, key):
        mdl = models.get(key)
        if mdl is None:
            return np.nan
        x = row[feat_prev_all].astype(float).fillna(0).to_frame().T
        return mdl.predict(x)[0]

    # stacking Ridge appris sur TRAIN
    tmp = df_t.loc[m_train].copy()
    G = [_pred_with_key(r, ("GLOBAL", "_")) for _, r in tmp.iterrows()]
    M = [_pred_with_key(r, (r["macro_pos"], "_")) for _, r in tmp.iterrows()]
    T = [_pred_with_key(r, (r["macro_pos"], r["tier"])) for _, r in tmp.iterrows()]
    X_stack = np.nan_to_num(np.c_[G, M, T], nan=0.0)
    y_stack = tmp["pts_target"].astype(float).values
    ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X_stack, y_stack)

    def predict_rows(sub):
        rows = []
        for _, r in sub.iterrows():
            g = _pred_with_key(r, ("GLOBAL", "_"))
            m = _pred_with_key(r, (r["macro_pos"], "_"))
            t = _pred_with_key(r, (r["macro_pos"], r["tier"]))
            rows.append([g, m, t])
        Xc = np.nan_to_num(np.array(rows), nan=0.0)
        return ridge.predict(Xc)

    def export_eval(mask, split_name):
        sub = df_t.loc[mask].copy()
        sub["y_pred"] = predict_rows(sub)
        m = metrics(sub["pts_target"].astype(float).values, sub["y_pred"].values)
        preds_cols = [
            "playerName",
            "team_prev",
            "pos_prev",
            "macro_pos",
            "tier",
            "seasonId",
            "pts_prev",
            "pts_target",
            "y_pred",
        ]
        preds_cols = [c for c in preds_cols if c in sub.columns]
        sub[preds_cols].to_csv(
            os.path.join(BASE_MODELS, f"predictions_{split_name}.csv"), index=False
        )
        pd.DataFrame([{"split": split_name, **m}]).to_csv(
            os.path.join(BASE_MODELS, f"metrics_{split_name}.csv"), index=False
        )
        print(f"[{split_name}] {m}")
        return m

    # Val/Test (on garde simple ici; rolling + HP mini peuvent être ajoutés ultérieurement si besoin)
    MET_VAL = export_eval(m_val, "val")
    MET_TEST = export_eval(m_test, "test")

    # next season (skip in fast mode)
    next_season = test_season + 101
    if not (args and args.fast):
        sub_next = df_t[df_t["seasonId"] == test_season].copy()
        if "age_prev" in sub_next.columns:
            sub_next["age_prev"] = pd.to_numeric(sub_next["age_prev"], errors="coerce") + 1
        for mp in ["FWD", "DEF"]:
            m_mp = sub_next["macro_pos"] == mp
            if tier_models.get(mp) is not None:
                km = tier_models[mp]
                labs = km.predict(sub_next.loc[m_mp, "pts_prev"].fillna(0).values.reshape(-1, 1))
                order = np.argsort(km.cluster_centers_.flatten())
                mapping = {
                    cl: name
                    for cl, name in zip(order, ["Low", "Mid", "High"][: len(order)], strict=False)
                }
                sub_next.loc[m_mp, "tier"] = [mapping[i] for i in labs]
            else:
                sub_next.loc[m_mp, "tier"] = "All"
        sub_next["y_pred"] = predict_rows(sub_next)
        out_cols = ["playerName", "team_prev", "pos_prev", "macro_pos", "tier", "pts_prev", "y_pred"]
        out_cols = [c for c in out_cols if c in sub_next.columns]
        sub_next["seasonId_pred"] = next_season
        sub_next[out_cols + ["seasonId_pred"]].to_csv(
            os.path.join(BASE_MODELS, f"predictions_next_{next_season}.csv"), index=False
        )

    # results.json pour CI
    results = {"val": MET_VAL, "test": MET_TEST, "splits": {"next": int(next_season)}}
    with open(os.path.join(BASE_MODELS, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    global RANDOM_STATE, args
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--fast", type=int, default=0)  # 1=rapide
    args = p.parse_args()
    RANDOM_STATE = args.seed

    df_raw = load_df()
    df = prepare_df(df_raw)
    print("[CLEAN] shape:", df.shape)
    res = run_training(df)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
