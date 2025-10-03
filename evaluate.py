# evaluate.py
import os, json, time, random, numpy as np, pandas as pd, yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    set_seed(cfg.get("seed", 42))
    Path("artifacts").mkdir(exist_ok=True)

    data_path = cfg["data"]["train_path"]
    target = cfg["data"]["target"]

    # 1) Charge l'Excel (1ère feuille)
    df = pd.read_excel(data_path, engine="openpyxl")

    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable dans {data_path}. Colonnes: {list(df.columns)[:15]} ...")

    # 2) Split simple (si tu as déjà un val set, remplace par deux reads)
    X = df.drop(columns=[target])
    y = df[target]

    # conversion de colonnes non numériques si nécessaire (naïf)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=cfg.get("seed", 42), stratify=y if y.nunique() > 1 else None
    )

    # 3) Train + eval
    start = time.time()
    model = LogisticRegression(max_iter=2000)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xva)
    elapsed = time.time() - start

    # 4) métriques macro (classif multi/binaire)
    f1 = f1_score(yva, yhat, average="macro")
    prec = precision_score(yva, yhat, average="macro", zero_division=0)
    rec = recall_score(yva, yhat, average="macro", zero_division=0)

    # 5) exports annexes (facultatifs mais pratiques)
    pd.DataFrame({"y_true": yva, "y_pred": yhat}).to_excel("artifacts/preds_val.xlsx", index=False)

    # 6) JSON standard pour l’agent
    results = {
        "objective": cfg.get("objective", "maximize:F1_macro"),
        "metrics": {"F1_macro": float(f1), "precision": float(prec), "recall": float(rec)},
        "train_time_s": float(elapsed),
        "params": {
            "lr": cfg["train"].get("lr"),
            "batch_size": cfg["train"].get("batch_size"),
            "weight_decay": cfg["train"].get("weight_decay"),
            "seed": cfg.get("seed"),
        },
    }
    with open("artifacts/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("OK — artifacts/results.json écrit.")

if __name__ == "__main__":
    main()
