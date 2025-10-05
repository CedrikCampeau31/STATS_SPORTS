# evaluate.py
import os, json, time, random, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")  # backend non interactif pour CI
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# --------- utils ----------
def set_seed(seed: int):
    import numpy as _np, random as _random
    _random.seed(seed); _np.random.seed(seed)

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def guess_target(df: pd.DataFrame) -> str | None:
    # Cherche des noms "classiques"
    candidates = [
        "label","target","y","class","Class","classe","is_top","isTop","isTop6",
        "PointsTier","Tier","Cluster","Outcome","Result","target_label"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Si rien: prend la DERNIÈRE colonne non-ID raisonnable
    id_like = {"id","ID","player_id","PlayerID","Name","Player","player","name"}
    cols = [c for c in df.columns if c not in id_like]
    return cols[-1] if cols else None

def is_classification(y: pd.Series) -> bool:
    # binaire/multi-classe si dtype non-numérique ou peu de valeurs distinctes
    if not pd.api.types.is_numeric_dtype(y):
        return True
    nunique = y.nunique(dropna=True)
    return nunique <= 20  # seuil souple pour NHL tiers/classes

def one_hot_safe(df: pd.DataFrame) -> pd.DataFrame:
    # cast object/category -> one-hot; garde numériques inchangés
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True, drop_first=False)

def save_confusion(y_true, y_pred, out_path):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# --------- main ----------
def main():
    cfg = load_cfg()
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    Path("artifacts").mkdir(exist_ok=True)

    data_path = cfg["data"]["train_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 1) read
    df = pd.read_excel(data_path, engine="openpyxl")
    if df.empty:
        raise ValueError("Dataset is empty.")

    # 2) pick target
    target = (cfg.get("data") or {}).get("target") or ""
    if not target:
        target = guess_target(df)
        if not target:
            raise ValueError("Impossible de déduire la colonne cible. Renseigne `data.target` dans config.yaml.")
        print(f"[info] target auto-détectée: {target}")

    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable. Colonnes disponibles: {list(df.columns)[:30]}...")

    # 3) X/y + prétraitements
    y = df[target]
    X = df.drop(columns=[target])

    # Remplacement simple des NA (num: médiane, cat: 'Unknown' via get_dummies)
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna("Unknown")

    X = one_hot_safe(X)

    # Scaling léger pour modèles linéaires
    scaler = None
    if is_classification(y):
        # classification -> on peut scaler pour LR
        scaler = StandardScaler(with_mean=False)  # sparse-friendly
        X_scaled = scaler.fit_transform(X)
    else:
        # si régression (peu probable ici), on scalerait aussi
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)

    # 4) split (stratify si classif)
    strat = y if is_classification(y) and y.nunique(dropna=True) > 1 else None
    Xtr, Xva, ytr, yva = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed, stratify=strat
    )

    # 5) modèle
    start = time.time()
    if is_classification(y):
        # Choix simple et robuste pour baseline multi-classe
        try:
            model = LogisticRegression(max_iter=4000, n_jobs=None)  # multi classe via one-vs-rest auto
        except TypeError:
            # (compat scikit)
            model = LogisticRegression(max_iter=4000)

        model.fit(Xtr, ytr)
        yhat = model.predict(Xva)
        f1 = f1_score(yva, yhat, average="macro")
        prec = precision_score(yva, yhat, average="macro", zero_division=0)
        rec = recall_score(yva, yhat, average="macro", zero_division=0)
    else:
        # (si régression un jour)
        raise NotImplementedError("Le script est paramétré pour la classification NHL (F1_macro).")

    elapsed = time.time() - start

    # 6) exports annexes
    pd.DataFrame({"y_true": yva, "y_pred": yhat}).to_excel("artifacts/preds_val.xlsx", index=False)
    save_confusion(yva, yhat, "artifacts/confusion_matrix.png")

    # 7) results.json (standard agent)
    results = {
        "objective": cfg.get("objective", "maximize:F1_macro"),
        "metrics": {"F1_macro": float(f1), "precision": float(prec), "recall": float(rec)},
        "train_time_s": float(elapsed),
        "params": {
            "lr": (cfg.get("train") or {}).get("lr"),
            "batch_size": (cfg.get("train") or {}).get("batch_size"),
            "weight_decay": (cfg.get("train") or {}).get("weight_decay"),
            "seed": seed,
        },
        "data": {
            "path": data_path,
            "n_samples": int(df.shape[0]),
            "n_features_after_encoding": int(X.shape[1])
        }
    }
    with open("artifacts/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("OK — artifacts/results.json écrit.")

if __name__ == "__main__":
    main()
