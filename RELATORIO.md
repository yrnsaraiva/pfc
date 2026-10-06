# Relatório – Modelo de risco de diabetes (STEPS Moçambique 2005/2014)

*Resultados de `metrics.json` (hold-out de 20 %). Reproduzíveis com `python train.py --build`.*

## 1. Resumo executivo

Foi construído um modelo de rastreio que estima o risco de diabetes **sem análise de sangue**, a partir de questionário e medições simples. Em dados nunca vistos, o ROC-AUC é de **0,70 a 0,73**: desempenho modesto, útil como filtro de rastreio e não como diagnóstico.

- **Dados:** inquéritos STEPS 2005 e 2014, harmonizados em 3 531 adultos de 25 a 64 anos com glicemia em jejum válida; 216 (6,1 %) com diabetes.
- **Alvo:** glicemia capilar em jejum ≥ 6,1 mmol/L ou tratamento com insulina/antidiabéticos orais, a definição de Madede et al. (2022), que analisa os mesmos inquéritos.
- **Melhor modelo:** regressão logística (ROC-AUC 0,733). No limiar de rastreio apanha 38 dos 43 doentes do teste (88 %), mas só 43 % dos saudáveis ficam sem alerta e só 9 % dos alertas são casos reais.
- **Diferença entre modelos:** os quatro estão dentro do ruído (43 positivos no teste); o modelo mais simples não fica atrás dos mais complexos.
- **Cautela principal:** as prevalências da amostra (3,2 % em 2005, 11,3 % em 2014) não são populacionais, e o desempenho cai para 0,62 a 0,68 de AUC quando se treina numa vaga e se testa na outra.

## 2. Dados e metodologia

### 2.1 Amostra

| Vaga | Amostras | Positivos | Prevalência (sem pesos) | Artigo, ponderada |
| --- | --- | --- | --- | --- |
| 2005 | 2 243 | 71 | 3,2 % | 2,9 % (n = 2 343) |
| 2014 | 1 288 | 145 | 11,3 % | 7,4 % (n = 1 321) |
| Total | 3 531 | 216 | 6,1 % | — |

Só entram adultos de 25 a 64 anos, não grávidas, com glicemia capilar em jejum de 12 h válida. Quem não cumpriu o jejum ou não tem glicemia fica de fora (não é tratado como negativo), como no artigo de referência. Idade em falta (215 casos, só em 2014) mantém-se e é imputada.

### 2.2 Alvo

`diabetes = 1` se glicemia ≥ 6,1 mmol/L (sangue total capilar, igual nas duas vagas) **ou** tratamento com insulina/antidiabéticos orais. A glicemia e o tratamento **não** são variáveis de entrada: com elas o modelo "copiava" o alvo.

### 2.3 Variáveis (20)

Idade, sexo, escolaridade, altura, peso, IMC, cintura, razão cintura/altura, tensão arterial (média das leituras, convenção STEPS), medicação e diagnóstico de hipertensão, tabaco (fumado e sem fumo), álcool (consumo e dias por mês), fruta e vegetais (porções/dia), actividade física (MET-min/semana, GPAQ) e tempo sedentário.

### 2.4 Treino e avaliação

1. 20 % dos dados (707 pessoas, 43 positivas) ficam de lado como teste, estratificados por vaga e alvo, **antes** de qualquer processamento.
2. Cada modelo é um pipeline: imputação pela mediana, normalização (regressão logística), classificador e calibração sigmoide.
3. O limiar de alerta é escolhido por validação cruzada (5 folds) no treino, para uma sensibilidade-alvo de 80 %.
4. Os dados **não foram balanceados** (prevalência de 6 % mantida): usam-se pesos de classe e calibração.
5. Os modelos finais são treinados com todos os dados e servidos por uma API (FastAPI).

Modelos: regressão logística (`lr`), Random Forest (`rf`), LightGBM (`lgbm`) e XGBoost (`xgb`).

## 3. Resultados

### 3.1 Hold-out (707 pessoas, 43 positivas, 664 negativas)

| Modelo | ROC-AUC | PR-AUC | Brier | Sensibilidade | Especificidade | Precisão | Limiar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Regressão logística (`lr`) | 0,733 | 0,167 | 0,055 | 0,88 | 0,43 | 0,092 | 0,042 |
| Random Forest (`rf`) | 0,721 | 0,154 | 0,055 | 0,77 | 0,44 | 0,082 | 0,045 |
| LightGBM (`lgbm`) | 0,696 | 0,126 | 0,056 | 0,84 | 0,44 | 0,088 | 0,046 |
| XGBoost (`xgb`) | 0,706 | 0,129 | 0,057 | 0,72 | 0,53 | 0,091 | 0,048 |

A validação cruzada no treino dá ROC-AUC de 0,70 a 0,71, consistente com o teste. A prevalência de referência é 6 %: um PR-AUC de 0,13 a 0,17 é cerca de 2 a 3 vezes melhor do que o acaso.

No limiar de rastreio, a regressão logística apanha 38 dos 43 doentes (5 escapam) e marca como risco 376 dos 664 saudáveis. A sensibilidade no teste (72 % a 88 %) varia à volta dos 80 % pretendidos porque há poucos casos.

### 3.2 Generalização entre vagas (ROC-AUC)

| Modelo | Treino 2005 → teste 2014 | Treino 2014 → teste 2005 |
| --- | --- | --- |
| `lr` | 0,671 | 0,679 |
| `rf` | 0,634 | 0,679 |
| `lgbm` | 0,619 | 0,649 |
| `xgb` | 0,629 | 0,661 |

## 4. Interpretação

- **ROC-AUC 0,73:** em cerca de 73 % dos pares (um doente, um saudável), o modelo dá risco maior ao doente. O acaso seria 0,5.
- **Sensibilidade 88 %, especificidade 43 %:** de cada 10 doentes, 9 são apanhados; de cada 10 saudáveis, quase 6 recebem alerta falso.
- **Precisão 9 %:** só cerca de 1 em cada 11 alertas é um caso real, porque a doença é rara. Subir o limiar reduz alertas falsos mas deixa escapar doentes.
- **Brier 0,055:** quase igual ao de um modelo que dissesse sempre 6 % (0,057). O ganho está na ordenação do risco e não em probabilidades muito precisas; o `risk_score` lê-se como estimativa grosseira.

**Variáveis que mais pesam** (regressão logística, variáveis normalizadas; associação, não causalidade):

- **Aumentam o risco:** idade (0,34), fruta (0,30), razão cintura/altura (0,24), dias de álcool por mês (0,23), hipertensão diagnosticada (0,19), escolaridade (0,17), cintura (0,16).
- **Diminuem o risco:** tabaco fumado (−0,24), tensão diastólica (−0,21), actividade física (−0,20), tabaco sem fumo (−0,14).

Idade e gordura abdominal são os sinais mais coerentes com a literatura. Efeitos de fruta, escolaridade e tabaco reflectem provavelmente factores de confusão (por exemplo, vida urbana e rendimento). Peso, IMC, cintura e razão cintura/altura são muito correlacionados, o que torna cada coeficiente individual instável.

## 5. Comparação com o artigo de referência

Madede et al., *BMC Public Health* 2022;22:2174, analisa os mesmos inquéritos com a definição "glicemia ≥ 6,1 mmol/L ou tratamento", só jejum de 12 h válido e 25–64 anos, com pesos amostrais.

- **2005:** com os dados brutos, ponderando, reproduzo 2,8 a 2,9 % (n = 2 338), contra 2,9 % no artigo.
- **2014:** a prevalência rural na amostra (6,7 %) aproxima-se da do artigo (6,1 %), mas a urbana é 19,6 % contra 10,2 %. Não consegui explicar a diferença: o CSV de 2014 não traz pesos e tem mais linhas (3 285) do que os participantes do artigo (3 119).
- **Glicemia média** sobe de 3,7 a 3,8 mmol/L em 2005 para 4,5 a 4,6 em 2014, também reportado pelo artigo.

Por isso as prevalências deste trabalho são da amostra, não da população.

## 6. Limitações

- **Poucos casos:** 216 positivos no total e 43 no teste. As diferenças entre modelos estão dentro do ruído.
- **Selecção por jejum:** só cerca de 60 % cumpriu o jejum, e esses eram mais velhos e mais rurais (viés reconhecido no artigo).
- **Sem pesos amostrais** no treino, e o CSV de 2014 não os fornece.
- **Colunas de 2014 inferidas:** não havia questionário de 2014. Altura, peso, cintura, tensão, jejum e tratamento foram identificados pela estrutura STEPS e pelos dados (por exemplo, a glicemia é mais alta em quem comeu). Devem ser confirmadas com a documentação oficial.
- **Idade em falta** em 215 casos de 2014, imputada pela mediana, o que enfraquece a variável mais importante.
- **Glicemia capilar** tende a subestimar a venosa.
- **Sem validação externa nem clínica:** o modelo não serve para diagnóstico.

## 7. Conclusões e próximos passos

Sem análise de sangue, idade e medidas corporais (sobretudo razão cintura/altura) bastam para uma triagem moderada, com ROC-AUC de cerca de 0,73.

- **Modelo recomendado:** regressão logística. Desempenho igual ou melhor que os de árvores, interpretável e a que melhor generaliza entre vagas.
- **Utilização:** primeiro filtro, enviando quem tiver risco ≥ 4,2 % para glicemia em jejum. Esperam-se muitos falsos alertas (cerca de 9 em 10 não têm diabetes).
- **Mudança face ao projecto anterior (BRFSS):** os ~97 % de acerto vinham de avaliar em dados reamostrados; os valores actuais são menores mas honestos.

Próximos passos:

1. Confirmar com a documentação de 2014 o significado das colunas inferidas e obter os pesos amostrais.
2. Testar o alvo com glicemia ≥ 7,0 mmol/L e comparar.
3. Avaliar variáveis hoje excluídas (região, urbano/rural, rendimento) e reduzir variáveis redundantes.
4. Validar com dados novos de Moçambique antes de qualquer uso prático.
