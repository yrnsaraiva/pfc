# Risco de diabetes – STEPS Moçambique (2005 + 2014)

Pipeline e API de rastreio de diabetes treinados com os inquéritos STEPS (OMS) de Moçambique.

## Fluxo
1. Colocar os CSV brutos em `data/raw/` (`STEPS2005.csv`, `STEPS2014.csv`; **não são versionados** – o ficheiro de 2014 contém nomes).
2. `python steps_data.py` → `data/processed/steps_mozambique_2005_2014.csv` (harmonizado, sem identificadores).
3. `python train.py` → `models/*.pkl`, `models/model_meta.json`, `metrics.json` (`--build` refaz o passo 2).
4. `uvicorn app:app --reload` → docs em `/api/v1/docs`.

## Harmonização e alvo (`steps_data.py`)
Alinhados com o artigo que analisa os mesmos inquéritos: Madede et al., *BMC Public Health* 2022;22:2174.
- 2005 usa códigos numéricos; 2014 usa texto ("SIM/NÃO/Sem informacao"). Ambos vão para o mesmo esquema de 20 features:
  idade, sexo, escolaridade, altura/peso/IMC, cintura e razão cintura/altura, TA sistólica/diastólica (média das leituras, convenção STEPS), medicação e diagnóstico de hipertensão, tabaco, álcool, fruta e vegetais, actividade física (MET-min/semana, GPAQ) e tempo sedentário.
- **Quem entra:** adultos de 25–64 anos com glicemia capilar em jejum (12 h) válida; grávidas excluídas. Quem não cumpriu o jejum ou não tem glicemia é **excluído** (não é tratado como negativo). Idade em falta (215 casos, só em 2014) mantém-se e é imputada.
- **Alvo:** glicemia em jejum ≥ 6,1 mmol/L (sangue total capilar, **igual nas duas vagas**) **ou** tratamento com insulina/antidiabéticos orais.
- Códigos de missing (77/88/99/999/"Sem informacao") e valores implausíveis → `NaN` (imputados no pipeline).
- A glicemia e as variáveis de tratamento **não** são features (evita fuga de informação): é um modelo de rastreio sem análise de sangue.

| Vaga | Amostras | Positivos | Prevalência (sem pesos) | Artigo (ponderada) |
| --- | --- | --- | --- | --- |
| 2005 | 2 243 | 71 | 3,2 % | 2,9 % (n = 2 343) |
| 2014 | 1 288 | 145 | 11,3 % | 7,4 % (n = 1 321) |

## Melhorias face à versão BRFSS (removido; histórico no git)
- Divisão treino/teste antes de imputar/escalar; tudo em `Pipeline` (sem fuga de informação). Antes, SMOTEENN+scaler eram aplicados antes do split e as métricas (~97 %) eram medidas em dados reamostrados.
- Classe rara (~5 %): ponderação de classes + calibração sigmoide (probabilidades reais), em vez de SMOTEENN.
- Métricas: ROC-AUC, PR-AUC, Brier, sensibilidade/especificidade num limiar escolhido por CV (sensibilidade-alvo 80 %), e validação entre vagas (2005→2014, 2014→2005).
- Novo modelo de base interpretável (regressão logística, `lr`).
- API: schema novo, campos opcionais imputados, `prediction` e níveis de risco derivados dos limiares do treino (`model_meta.json`).

## Resultados (hold-out 20 %, n = 707, 43 positivos)
Ver `metrics.json`. ROC-AUC 0,70–0,73 (melhor: regressão logística, 0,733); PR-AUC 0,13–0,17 (prevalência 6 %); entre vagas, ROC-AUC 0,62–0,68. Valores realistas para rastreio sem sangue – os ~97 % do projecto BRFSS eram artefacto da avaliação.

## Limitações / pressupostos a validar
- Questionário 2014 não foi fornecido: o significado das colunas (m3_a–f = TA 3 leituras, m7_a/m7_p = altura em m/peso, m9 = cintura, b1 = comeu/bebeu nas últimas 12 h, b4 = glicemia, b5/h8/h9 = tratamento) foi inferido da estrutura STEPS e dos dados.
- O CSV de 2014 não traz pesos amostrais e não reproduz exactamente a prevalência do artigo (sobretudo em áreas urbanas: 19,6 % vs 10,2 %); as prevalências aqui são da amostra, não da população. Pesos de 2005 (`analysisweight`) também não são usados no treino.
- Como no artigo, só quem cumpriu o jejum entra (cerca de 60 %); esses eram mais velhos e mais rurais, o que pode enviesar.
- Glicemia capilar subestima a venosa; a glicemia média sobe de 3,7 (2005) para 4,5 mmol/L (2014).
- Idade em falta em 215 casos de 2014 (imputada pela mediana) enfraquece o sinal da variável mais importante.
- Poucos positivos (216): intervalos de confiança largos; resultados indicativos, não para uso clínico.
