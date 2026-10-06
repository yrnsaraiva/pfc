"""
Pré-processamento e harmonização dos inquéritos STEPS (OMS) de Moçambique
(2005 e 2014) para o problema de rastreio/risco de diabetes.

Uso (linha de comandos):
    python steps_data.py --raw-dir data/raw --out data/processed/steps_mozambique_2005_2014.csv

Principais decisões (ver README):
- As duas vagas têm codificações diferentes (2005: códigos numéricos; 2014: texto em
  português com "Sem informacao"). Tudo é harmonizado para o mesmo esquema de colunas.
- Definição alinhada com Madede et al., BMC Public Health 2022;22:2174 (mesmos inquéritos):
  adultos de 25-64 anos com glicemia capilar em jejum (12 h) válida; `diabetes` = 1 se
  glicemia >= GLUCOSE_THRESHOLD (6,1 mmol/L, sangue total capilar, igual nas duas vagas)
  OU em tratamento com insulina/antidiabéticos orais. Quem não cumpriu o jejum ou não tem
  glicemia é excluído (não é tratado como negativo).
- A glicemia e as variáveis de tratamento NÃO são features (evitar fuga de informação):
  o modelo é uma ferramenta de rastreio sem análise sanguínea.
- Identificadores pessoais (nome, local, ids) nunca são escritos no ficheiro processado.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# Glicemia capilar em jejum (mmol/L). WHO STEPS: >= 6.1 (sangue total capilar) = glicemia elevada;
# 7.0 corresponde ao critério em plasma venoso. Configurável.
GLUCOSE_THRESHOLD = 6.1
GLUCOSE_MAX_VALID = 33.3  # limite superior típico dos glucómetros; acima disso = erro/código

FEATURES: List[str] = [
    "age",
    "sex_male",
    "education_years",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "waist_height_ratio",
    "sbp",
    "dbp",
    "bp_meds",
    "told_hypertension",
    "smoker_current",
    "smokeless_current",
    "alcohol_past12m",
    "alcohol_days_month",
    "fruit_servings_day",
    "veg_servings_day",
    "met_min_week",
    "sedentary_hours_day",
]
TARGET = "diabetes"
META_COLS = ["survey_year", "glucose_mmol", "fasted", "on_treatment"]
AGE_MIN, AGE_MAX = 25, 64  # faixa etária do STEPS Moçambique


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def num(s: pd.Series, missing: Iterable[float] = ()) -> pd.Series:
    """Converte para numérico ('Sem informacao' etc. -> NaN) e anula códigos de missing."""
    out = pd.to_numeric(s, errors="coerce")
    return out.mask(out.isin(list(missing)))


def in_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.where((s >= lo) & (s <= hi))


def yes_no(s: pd.Series, yes, no) -> pd.Series:
    """Mapeia para 1/0; qualquer outro valor (inclui 'Sem informacao', 9) -> NaN."""
    out = pd.Series(np.nan, index=s.index)
    out[s.isin(yes if isinstance(yes, (list, set, tuple)) else [yes])] = 1.0
    out[s.isin(no if isinstance(no, (list, set, tuple)) else [no])] = 0.0
    return out


def mean_bp(sys_cols: List[pd.Series], dia_cols: List[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """
    Convenção STEPS: com 3 leituras usa-se a média da 2ª e 3ª; com 2 leituras a média das duas;
    com 1 leitura, essa leitura. Leituras pareadas (sistólica/diastólica válidas em simultâneo).
    """
    sys_df = pd.concat(sys_cols, axis=1)
    dia_df = pd.concat(dia_cols, axis=1)
    valid = sys_df.notna().values & dia_df.notna().values
    sys_df = sys_df.where(valid)
    dia_df = dia_df.where(valid)
    n = valid.sum(axis=1)
    use_last_two = n >= 3
    # remove a 1ª leitura quando há 3 válidas
    sys_vals = sys_df.copy()
    dia_vals = dia_df.copy()
    sys_vals.loc[use_last_two, sys_vals.columns[0]] = np.nan
    dia_vals.loc[use_last_two, dia_vals.columns[0]] = np.nan
    return sys_vals.mean(axis=1), dia_vals.mean(axis=1)


def minutes(h: pd.Series, m: pd.Series) -> pd.Series:
    mins = h * 60 + m
    return mins.where((h >= 0) & (h <= 16) & (m >= 0) & (m < 60) & (mins <= 16 * 60))


def pa_domain(flag: pd.Series, days: pd.Series, mins_day: pd.Series, met: float) -> pd.Series:
    """MET-min/semana de um domínio GPAQ. Resposta 'não' => 0; 'sim' com dados inválidos => NaN."""
    days = in_range(days, 0, 7)
    val = met * days * mins_day
    val = val.where(flag == 1)
    val[flag == 0] = 0.0
    return val


def finalize(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    df["bmi"] = in_range(df["bmi"], 12, 70)
    df["waist_height_ratio"] = df["waist_cm"] / df["height_cm"]
    df["survey_year"] = year
    return df[FEATURES + META_COLS + [TARGET]]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gl = df["glucose_mmol"].where(df["fasted"] == 1)
    treated = df["on_treatment"] == 1
    df[TARGET] = ((gl >= GLUCOSE_THRESHOLD) | treated).astype(float).where(gl.notna())
    return df


def in_age_range(age: pd.Series) -> pd.Series:
    """25-64 anos; idade em falta mantém-se (será imputada), como em Madede et al."""
    return age.isna() | age.between(AGE_MIN, AGE_MAX)


# -----------------------------------------------------------------------------
# STEPS 2005 (códigos numéricos)
# -----------------------------------------------------------------------------
def load_2005(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, low_memory=False)
    d = pd.DataFrame(index=r.index)

    d["age"] = num(r["age"])
    d["sex_male"] = yes_no(r["sex"], "Men", "Women")
    d["education_years"] = in_range(num(r["c4"], [77, 88, 99]), 0, 25)

    d["height_cm"] = in_range(num(r["m3"], [999.9]), 100, 230)
    d["weight_kg"] = in_range(num(r["m4"], [999.9]), 25, 250)
    d["waist_cm"] = in_range(num(r["m7"], [999.9]), 40, 200)
    pregnant = num(r["m5"]) == 1

    sys_cols = [in_range(num(r[c], [999]), 70, 260) for c in ("m11a", "m12a", "m13a")]
    dia_cols = [in_range(num(r[c], [999]), 40, 150) for c in ("m11b", "m12b", "m13b")]
    d["sbp"], d["dbp"] = mean_bp(sys_cols, dia_cols)
    d["bp_meds"] = yes_no(num(r["m14"]), 1, 2)
    d["told_hypertension"] = yes_no(num(r["h2"]), 1, 2).fillna(0.0)

    d["smoker_current"] = yes_no(num(r["t1"]), 1, 2)
    d["smokeless_current"] = yes_no(num(r["t9"]), 1, 2)

    # a1: bebeu nos últimos 12 meses; a2: frequência (1: >=5 d/sem ... 4: <1 vez/mês)
    d["alcohol_past12m"] = yes_no(num(r["a1"]), 1, 2)
    freq_map = {1: 24.0, 2: 10.0, 3: 2.0, 4: 0.5}
    d["alcohol_days_month"] = num(r["a2"]).map(freq_map)
    d.loc[d["alcohol_past12m"] == 0, "alcohol_days_month"] = 0.0

    # Fruta/vegetais: dias/semana x porções/dia / 7
    fruit_days = in_range(num(r["d1"], [77, 88, 99]), 0, 7)
    fruit_serv = in_range(num(r["d2"], [77, 88, 99]), 0, 15)
    veg_days = in_range(num(r["d3"], [77, 88, 99]), 0, 7)
    veg_serv = in_range(num(r["d4"], [77, 88, 99]), 0, 15)
    d["fruit_servings_day"] = (fruit_days * fruit_serv / 7).where(fruit_days > 0, fruit_days)
    d["veg_servings_day"] = (veg_days * veg_serv / 7).where(veg_days > 0, veg_days)

    # Actividade física (GPAQ): trabalho vigoroso/moderado, deslocação, lazer vigoroso/moderado
    def t(h, m):
        return minutes(num(r[h], [77, 99]), num(r[m], [77, 99]))

    flag = lambda c: yes_no(num(r[c]), 1, 2)  # noqa: E731
    domains = [
        pa_domain(flag("p1"), num(r["p2"], [77, 99]), t("p3a", "p3b"), 8),
        pa_domain(flag("p4"), num(r["p5"], [77, 99]), t("p6a", "p6b"), 4),
        pa_domain(flag("p7"), num(r["p8"], [77, 99]), t("p9a", "p9b"), 4),
        pa_domain(flag("p10"), num(r["p11"], [77, 99]), t("p12a", "p12b"), 8),
        pa_domain(flag("p13"), num(r["p14"], [77, 99]), t("p15a", "p15b"), 4),
    ]
    d["met_min_week"] = pd.concat(domains, axis=1).sum(axis=1, min_count=1)
    sed_h = num(r["p16a"], [77, 99])
    sed_m = num(r["p16b"], [77, 99])
    d["sedentary_hours_day"] = (minutes(sed_h, sed_m) / 60).where(lambda s: s > 0)

    # --- alvo ---
    d["glucose_mmol"] = in_range(num(r["b5"], [999]), 0.5, GLUCOSE_MAX_VALID)
    d["fasted"] = yes_no(num(r["b1"]), 2, 1)  # b1: 1 = comeu/bebeu nas últimas 12 h
    # tratamento: h8a = insulina, h8b = antidiabéticos orais (perguntados a quem tem diagnóstico)
    d["on_treatment"] = ((num(r["h8a"]) == 1) | (num(r["h8b"]) == 1)).astype(float)

    d = d[~pregnant & in_age_range(d["age"])]
    d = add_target(d)
    return finalize(d, 2005)


# -----------------------------------------------------------------------------
# STEPS 2014 (texto em português)
# -----------------------------------------------------------------------------
def load_2014(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, low_memory=False)
    # inquiridos sem entrevista (todas as secções vazias)
    r = r[r["c_0"] == "Sim"].copy()
    d = pd.DataFrame(index=r.index)

    d["age"] = num(r["c3_IDADE REC"], [99, 999])
    d["sex_male"] = yes_no(r["c1"], "Masculino", "Feminino")
    d["education_years"] = in_range(num(r["c4"], [77, 88, 99]), 0, 25)

    d["height_cm"] = in_range(num(r["m7_a"], [9.99]) * 100, 100, 230)
    d["weight_kg"] = in_range(num(r["m7_p"], [999.9]), 25, 250)
    d["waist_cm"] = in_range(num(r["m9"], [999.9]), 40, 200)
    pregnant = r["m5"] == "SIM"

    # m3_a/b, m3_c/d, m3_e/f = sistólica/diastólica das leituras 1, 2 e 3
    sys_cols = [in_range(num(r[c], [999]), 70, 260) for c in ("m3_a", "m3_c", "m3_e")]
    dia_cols = [in_range(num(r[c], [999]), 40, 150) for c in ("m3_b", "m3_d", "m3_f")]
    d["sbp"], d["dbp"] = mean_bp(sys_cols, dia_cols)
    d["bp_meds"] = yes_no(r["m4"], "SIM", "NÃO")
    d["told_hypertension"] = yes_no(r["h2"], "SIM", "NÃO").fillna(0.0)

    d["smoker_current"] = yes_no(r["t1"], "SIM", "NÃO")
    d["smokeless_current"] = yes_no(r["t12"], "SIM", "NÃO")

    d["alcohol_past12m"] = yes_no(r["a_2"], "SIM", "NÃO")
    d.loc[r["a_1"] == "NÃO", "alcohol_past12m"] = 0.0  # nunca bebeu
    freq_map = {
        "Todos os dias": 30.0,
        "5 a 6 dias por semana": 22.0,
        "3 a 4 dias por semana": 14.0,
        "1 a 2 dias por semana": 6.0,
        "1 a 3 dias por mês": 2.0,
        "Menos que uma vez por mês": 0.5,
    }
    d["alcohol_days_month"] = r["a_4"].map(freq_map)
    d.loc[d["alcohol_past12m"] == 0, "alcohol_days_month"] = 0.0

    fruit_days = in_range(num(r["d1"], [77, 88, 99]), 0, 7)
    fruit_serv = in_range(num(r["d2"], [77, 88, 99]), 0, 15)
    veg_days = in_range(num(r["d3"], [77, 88, 99]), 0, 7)
    veg_serv = in_range(num(r["d4"], [77, 88, 99]), 0, 15)
    d["fruit_servings_day"] = (fruit_days * fruit_serv / 7).where(fruit_days > 0, fruit_days)
    d["veg_servings_day"] = (veg_days * veg_serv / 7).where(veg_days > 0, veg_days)

    def t(h, m):
        return minutes(num(r[h], [77, 99]), num(r[m], [77, 99]))

    flag = lambda c: yes_no(r[c], "SIM", "NÃO")  # noqa: E731
    domains = [
        pa_domain(flag("p1"), num(r["p2"], [77, 99]), t("p3_h", "p3_m"), 8),
        pa_domain(flag("p4"), num(r["p5"], [77, 99]), t("p6_h", "p6_m"), 4),
        pa_domain(flag("p7"), num(r["p8"], [77, 99]), t("p9_h", "p9_m"), 4),
        pa_domain(flag("p10"), num(r["p11"], [77, 99]), t("p12_h", "p12_m"), 8),
        pa_domain(flag("p13"), num(r["p14"], [77, 99]), t("p15_h", "p15_m"), 4),
    ]
    d["met_min_week"] = pd.concat(domains, axis=1).sum(axis=1, min_count=1)
    sed = minutes(num(r["p16_h"], [77, 99]), num(r["p16_m"], [77, 99]))
    d["sedentary_hours_day"] = (sed / 60).where(lambda s: s > 0)

    # --- alvo ---
    d["glucose_mmol"] = in_range(num(r["b4"], [99.9]), 0.5, GLUCOSE_MAX_VALID)
    d["fasted"] = yes_no(r["b1"], "NÃO", "SIM")  # b1: comeu/bebeu nas últimas 12 h?
    # tratamento: b5 = medicação para glicemia (passo 3); h8/h9 = insulina/orais (a confirmar)
    d["on_treatment"] = ((r["b5"] == "SIM") | (r["h8"] == "SIM") | (r["h9"] == "SIM")).astype(float)

    d = d[~pregnant & in_age_range(d["age"])]
    d = add_target(d)
    return finalize(d, 2014)


# -----------------------------------------------------------------------------
def build_dataset(raw_dir: Path) -> pd.DataFrame:
    f05 = next(raw_dir.glob("*2005*.csv"))
    f14 = next(raw_dir.glob("*2014*.csv"))
    df = pd.concat([load_2005(f05), load_2014(f14)], ignore_index=True)
    return df[df[TARGET].notna()].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/steps_mozambique_2005_2014.csv")
    args = ap.parse_args()

    df = build_dataset(Path(args.raw_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} linhas -> {out}")
    print(df.groupby("survey_year")[TARGET].agg(["size", "sum", "mean"]))


if __name__ == "__main__":
    main()
