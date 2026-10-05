# Risco de diabetes – STEPS Moçambique (2005 + 2014)

Pipeline e API de rastreio de diabetes treinados com os inquéritos STEPS (OMS) de Moçambique.

## Fluxo
1. Colocar os CSV brutos em `data/raw/` (`STEPS2005.csv`, `STEPS2014.csv`; **não são versionados** – o ficheiro de 2014 contém nomes).
2. `python steps_data.py` → `data/processed/steps_mozambique_2005_2014.csv` (harmonizado, sem identificadores).
3. `python train.py` → `models/*.pkl`, `models/model_meta.json`, `metrics.json` (`--build` refaz o passo 2).
4. `uvicorn app:app --reload` → docs em `/api/v1/docs`.

## Harmonização (`steps_data.py`)
- 2005 usa códigos numéricos; 2014 usa texto ("SIM/NÃO/Sem informacao"). Ambos vão para o mesmo esquema de 20 features:
  idade, sexo, escolaridade, altura/peso/IMC, cintura e razão cintura/altura, TA sistólica/diastólica (convenção STEPS: média das 2ª e 3ª leituras),
  medicação e diagnóstico de hipertensão, tabaco (fumado / sem fumo), álcool, fruta e vegetais (porções/dia), actividade física (MET-min/semana, GPAQ) e tempo sedentário.
- Grávidas e menores de 18 anos excluídos; códigos de missing (77/88/99/999/“Sem informacao”) e valores implausíveis → `NaN` (imputados no pipeline).
- **Alvo**: glicemia capilar em jejum ≥ 6,1 mmol/L (critério OMS STEPS; configurável em `GLUCOSE_THRESHOLD`) **ou** diagnóstico médico de diabetes. Só amostras com glicemia em jejum válida ou diagnóstico entram.
- A glicemia e as variáveis de diagnóstico/tratamento **não** são features (evita fuga de informação): é um modelo de rastreio sem análise de sangue.

## Melhorias face à versão BRFSS (`legacy_brfss/`)
- Divisão treino/teste antes de imputar/escalar; tudo em `Pipeline` (sem fuga de informação). Antes, SMOTEENN+scaler eram aplicados antes do split e as métricas (~97 %) eram medidas em dados reamostrados.
- Classe rara (~5 %): ponderação de classes + calibração sigmoide (probabilidades reais), em vez de SMOTEENN.
- Métricas: ROC-AUC, PR-AUC, Brier, sensibilidade/especificidade num limiar escolhido por CV (sensibilidade-alvo 80 %), e validação entre vagas (2005→2014, 2014→2005).
- Novo modelo de base interpretável (regressão logística, `lr`).
- API: schema novo, campos opcionais imputados, `prediction` e níveis de risco derivados dos limiares do treino (`model_meta.json`).

## Resultados (hold-out 20 %, n≈977)
Ver `metrics.json`. Ordem de grandeza: ROC-AUC ≈ 0,74–0,76; PR-AUC ≈ 0,15–0,23 (prevalência 5 %); entre vagas, ROC-AUC ≈ 0,62–0,71. São valores realistas para rastreio sem sangue – os ~97 % anteriores eram artefacto da avaliação.

## Limitações / pressupostos a validar
- Questionário 2014 não foi fornecido: o significado das colunas (m3_a–f = TA 3 leituras, m7_a/m7_p = altura em m/peso, m9 = cintura, b1 = comeu/bebeu nas últimas 12 h, b4 = glicemia, h7 = diagnóstico de diabetes, etc.) foi inferido da estrutura STEPS e dos dados (ex.: glicemia mais alta em quem comeu confirma b1).
- As medições de glicemia diferem entre vagas (mediana 3,7 vs 4,7 mmol/L; prevalência 2,7 % vs 9,3 %) – provável diferença de aparelho/calibração; daí o desempenho inferior entre vagas.
- `regiao` (2005) não tem legenda e 2014 só tem província/urbano-rural, por isso região/área não é usada. Pesos amostrais (`analysisweight`, 2005) não são usados.
- Poucos positivos (242): intervalos de confiança largos; resultados indicativos, não para uso clínico.
