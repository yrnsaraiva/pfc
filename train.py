"""
Treino e avaliação dos modelos de risco de diabetes (STEPS Moçambique 2005 + 2014).

    python train.py                       # usa data/processed/steps_mozambique_2005_2014.csv
    python train.py --build               # reconstrói o dataset a partir de data/raw primeiro

Melhorias face à versão BRFSS:
- Sem fuga de informação: separação treino/teste ANTES de qualquer imputação/escala/reamostragem
  (tudo dentro de Pipelines) e métricas reportadas apenas em dados nunca vistos.
- Classe rara (~5%): usa-se ponderação de classes + calibração de probabilidades em vez de SMOTEENN
  (que inflaciona métricas quando avaliado em dados reamostrados).
- Métricas adequadas a dados desbalanceados: ROC-AUC, PR-AUC, Brier, sensibilidade/especificidade
  num limiar de rastreio escolhido por validação cruzada.
- Validação entre vagas (treina 2005 -> testa 2014 e vice-versa) para medir generalização.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from steps_data import FEATURES, GLUCOSE_THRESHOLD, TARGET, build_dataset

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = Path("data/processed/steps_mozambique_2005_2014.csv")
MODELS_DIR = Path("models")
TARGET_SENSITIVITY = 0.80  # rastreio: prioriza apanhar casos


def make_models() -> dict:
    imputer = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "lr": Pipeline([
            ("imputer", imputer()),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
        ]),
        "rf": Pipeline([
            ("imputer", imputer()),
            ("model", RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        ]),
        "lgbm": Pipeline([
            ("imputer", imputer()),
            ("model", LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                class_weight="balanced", random_state=SEED, verbose=-1)),
        ]),
        "xgb": Pipeline([
            ("imputer", imputer()),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                eval_metric="logloss", random_state=SEED, n_jobs=-1)),
        ]),
    }


def calibrated(pipe: Pipeline) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=5)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Maior limiar que ainda atinge a sensibilidade-alvo."""
    best = float(p.min())
    for t in np.unique(np.round(p, 4)):
        if (p[y == 1] >= t).mean() >= target:
            best = float(t)
    return best


def rates(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return {
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def evaluate(y, p, t=None) -> dict:
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "prevalence": float(np.mean(y)),
    }
    if t is not None:
        out["threshold"] = float(t)
        out.update(rates(np.asarray(y), np.asarray(p), t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="reconstrói o dataset de data/raw")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--data", default=str(DATA_PATH))
    args = ap.parse_args()

    data_path = Path(args.data)
    if args.build or not data_path.exists():
        df = build_dataset(Path(args.raw_dir))
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    df = pd.read_csv(data_path)

    X, y, year = df[FEATURES], df[TARGET].astype(int), df["survey_year"]
    print(f"{len(df)} amostras | prevalência {y.mean():.3f} | positivos {y.sum()}")

    # Hold-out final (estratificado por vaga x alvo)
    strat = year.astype(str) + "_" + y.astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=SEED)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    metrics, meta_models = {}, {}
    MODELS_DIR.mkdir(exist_ok=True)

    for name, pipe in make_models().items():
        # 1) probabilidades fora-da-amostra (CV) no treino -> limiar de rastreio e métricas CV
        oof = cross_val_predict(calibrated(pipe), X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
        thr = threshold_for_sensitivity(y_tr.values, oof, TARGET_SENSITIVITY)
        cv_metrics = evaluate(y_tr.values, oof, thr)

        # 2) hold-out nunca visto
        model_tr = calibrated(pipe).fit(X_tr, y_tr)
        p_te = model_tr.predict_proba(X_te)[:, 1]
        test_metrics = evaluate(y_te.values, p_te, thr)

        # 3) generalização entre vagas
        cross = {}
        for src, dst in ((2005, 2014), (2014, 2005)):
            m = calibrated(pipe).fit(X[year == src], y[year == src])
            p = m.predict_proba(X[year == dst])[:, 1]
            cross[f"train{src}_test{dst}"] = {
                "roc_auc": float(roc_auc_score(y[year == dst], p)),
                "pr_auc": float(average_precision_score(y[year == dst], p)),
            }

        # 4) modelo final: treinado com todos os dados
        final = calibrated(pipe).fit(X, y)
        joblib.dump(final, MODELS_DIR / f"pipeline_{name}.pkl")

        metrics[name] = {"cv_train": cv_metrics, "holdout_test": test_metrics, "cross_wave": cross}
        meta_models[name] = {
            "screening_threshold": thr,
            "high_risk_threshold": float(min(max(2 * thr, thr + 0.05), 0.95)),
        }
        print(f"{name:5s} CV AUC={cv_metrics['roc_auc']:.3f} | test AUC={test_metrics['roc_auc']:.3f} "
              f"PR-AUC={test_metrics['pr_auc']:.3f} sens={test_metrics['sensitivity']:.2f} "
              f"spec={test_metrics['specificity']:.2f} | cross {cross}")

    (MODELS_DIR / "model_meta.json").write_text(json.dumps({
        "features": FEATURES,
        "target": TARGET,
        "glucose_threshold_mmol_l": GLUCOSE_THRESHOLD,
        "target_sensitivity": TARGET_SENSITIVITY,
        "n_samples": int(len(df)),
        "prevalence": float(y.mean()),
        "models": meta_models,
    }, indent=2), encoding="utf-8")
    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Modelos em models/, métricas em metrics.json")


if __name__ == "__main__":
    main()
