"""
API de Avaliação de Risco de Diabetes (STEPS Moçambique 2005/2014) - FastAPI

Boas práticas incluídas:
- risk_score + risk_level (thresholds configuráveis)
- Endpoints: health, models, model detail, predict, predict/batch, metrics, model health
- Erros padronizados (ErrorResponse) + logging consistente
- Carregamento de modelos com tratamento de erro e metadata (model_id/model_version)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, conint, confloat

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
API_PREFIX = "/api/v1"

logger = logging.getLogger("diabetes_api")
logging.basicConfig(level=logging.INFO)

# Ajuste paths conforme teu projeto
MODELS_DIR = Path("models")
METRICS_PATH = Path("metrics.json")
STATIC_DIR = Path(__file__).parent / "static"
META_PATH = MODELS_DIR / "model_meta.json"

# Registry de modelos (inclui metadata para rastreabilidade)
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "lr": {
        "pipeline_path": str(MODELS_DIR / "pipeline_lr.pkl"),
        "model_id": "diabetes-risk-lr",
        "model_version": "2.0.0",
    },
    "rf": {
        "pipeline_path": str(MODELS_DIR / "pipeline_rf.pkl"),
        "model_id": "diabetes-risk-rf",
        "model_version": "2.0.0",
    },
    "lgbm": {
        "pipeline_path": str(MODELS_DIR / "pipeline_lgbm.pkl"),
        "model_id": "diabetes-risk-lgbm",
        "model_version": "2.0.0",
    },
    "xgb": {
        "pipeline_path": str(MODELS_DIR / "pipeline_xgb.pkl"),
        "model_id": "diabetes-risk-xgb",
        "model_version": "2.0.0",
    },
}


def load_meta() -> Dict[str, Any]:
    if not META_PATH.exists():
        raise RuntimeError(f"Metadata de treino não encontrada em {META_PATH}. Execute train.py")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


META = load_meta()
FEATURE_ORDER: List[str] = META["features"]


def thresholds_for(model_name: str) -> Dict[str, float]:
    """Limiares calibrados no treino: screening (sensibilidade-alvo) e alto risco."""
    m = META["models"][model_name]
    return {"low": m["screening_threshold"], "high": m["high_risk_threshold"]}


ModelName = Literal["lr", "rf", "lgbm", "xgb"]
RiskLevel = Literal["low", "medium", "high"]


# -----------------------------------------------------------------------------
# Schemas (Pydantic)
# -----------------------------------------------------------------------------
class ErrorInner(BaseModel):
    code: str
    message: str
    details: Optional[List[Dict[str, Any]]] = None


class ErrorResponse(BaseModel):
    error: ErrorInner


class Features(BaseModel):
    """
    Features do rastreio STEPS (OMS) - apenas dados de questionário e medições não invasivas.
    Campos opcionais em falta são imputados (mediana do treino) pelo pipeline.
    """

    age: conint(ge=25, le=64) = Field(..., description="Idade em anos (o modelo foi treinado com 25-64)")
    sex_male: conint(ge=0, le=1) = Field(..., description="1 = masculino, 0 = feminino")

    # antropometria
    height_cm: Optional[confloat(ge=100, le=230)] = Field(None, description="Altura (cm)")
    weight_kg: Optional[confloat(ge=25, le=250)] = Field(None, description="Peso (kg)")
    waist_cm: Optional[confloat(ge=40, le=200)] = Field(None, description="Perímetro da cintura (cm)")

    # tensão arterial
    sbp: Optional[confloat(ge=70, le=260)] = Field(None, description="Tensão sistólica (mmHg)")
    dbp: Optional[confloat(ge=40, le=150)] = Field(None, description="Tensão diastólica (mmHg)")
    bp_meds: Optional[conint(ge=0, le=1)] = Field(None, description="Toma medicação para hipertensão")
    told_hypertension: conint(ge=0, le=1) = Field(0, description="Já lhe disseram que tem hipertensão")

    # estilo de vida
    education_years: Optional[confloat(ge=0, le=25)] = Field(None, description="Anos de escolaridade")
    smoker_current: conint(ge=0, le=1) = Field(0, description="Fuma actualmente")
    smokeless_current: conint(ge=0, le=1) = Field(0, description="Usa tabaco sem fumo")
    alcohol_past12m: conint(ge=0, le=1) = Field(0, description="Bebeu álcool nos últimos 12 meses")
    alcohol_days_month: Optional[confloat(ge=0, le=31)] = Field(
        None, description="Dias de consumo de álcool por mês (aprox.)")
    fruit_servings_day: Optional[confloat(ge=0, le=15)] = Field(None, description="Porções de fruta/dia")
    veg_servings_day: Optional[confloat(ge=0, le=15)] = Field(None, description="Porções de vegetais/dia")
    met_min_week: Optional[confloat(ge=0, le=60000)] = Field(
        None, description="Actividade física (MET-min/semana, GPAQ)")
    sedentary_hours_day: Optional[confloat(ge=0, le=16)] = Field(None, description="Horas sentado/dia")

    def to_frame(self) -> pd.DataFrame:
        """DataFrame com as colunas (incl. derivadas) pela ordem usada no treino."""
        row = self.model_dump()
        h, w, wc = row["height_cm"], row["weight_kg"], row["waist_cm"]
        row["bmi"] = w / (h / 100) ** 2 if h and w else None
        row["waist_height_ratio"] = wc / h if wc and h else None
        return pd.DataFrame([row], columns=FEATURE_ORDER).astype(float)


class PredictRequest(BaseModel):
    model_name: ModelName = Field(..., description="Modelo a usar: lr|rf|lgbm|xgb")
    subject_id: Optional[str] = Field(None, description="ID do utente/participante (opcional)")
    features: Features
    explain: bool = Field(False, description="Se true, devolve o efeito de cada grupo de variáveis no risco")
    persist: bool = Field(False, description="Reservado para futuro (auditoria/armazenamento)")


class ModelRef(BaseModel):
    name: ModelName
    model_id: str
    model_version: str


class PredictResult(BaseModel):
    prediction: Literal[0, 1]
    risk_score: confloat(ge=0, le=1)
    risk_level: RiskLevel


class FactorEffect(BaseModel):
    key: str
    label: str
    effect: float = Field(..., description="Variação do risco (pontos percentuais, em fracção 0-1) face à pessoa de referência")
    provided: bool = Field(..., description="False se o utilizador não indicou nenhum dos campos do grupo")
    features: List[str]


class Explanation(BaseModel):
    method: str
    baseline_score: float = Field(..., description="Risco da pessoa de referência (medianas da amostra de treino)")
    factors: List[FactorEffect]
    note: str


class PredictResponse(BaseModel):
    request_id: str
    model: ModelRef
    subject_id: Optional[str]
    result: PredictResult
    explanation: Optional[Explanation] = None
    created_at: str


class BatchItem(BaseModel):
    subject_id: Optional[str] = None
    features: Features


class BatchPredictRequest(BaseModel):
    model_name: ModelName
    items: List[BatchItem] = Field(..., min_length=1)
    persist: bool = False


class BatchPredictResult(BaseModel):
    subject_id: Optional[str]
    prediction: Literal[0, 1]
    risk_score: confloat(ge=0, le=1)
    risk_level: RiskLevel


class BatchPredictResponse(BaseModel):
    batch_id: str
    model: ModelRef
    results: List[BatchPredictResult]
    created_at: str


class HealthResponse(BaseModel):
    status: str
    time: str


class ModelSummary(BaseModel):
    name: ModelName
    model_id: str
    model_version: str
    has_predict_proba: bool


class ModelsListResponse(BaseModel):
    available_models: List[ModelSummary]
    total_models: int


class ModelInfoResponse(BaseModel):
    name: ModelName
    model_id: str
    model_version: str
    thresholds: Dict[str, float]
    glucose_threshold_mmol_l: float
    features: List[str]


class MetricsResponse(BaseModel):
    model: str
    metrics: Dict[str, Any]


class ModelHealthResponse(BaseModel):
    model: ModelName
    status: str
    has_predict_proba: bool
    model_type: str


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def error(code: str, message: str, details: Optional[List[Dict[str, Any]]] = None) -> HTTPException:
    payload = ErrorResponse(error=ErrorInner(code=code, message=message, details=details)).model_dump()
    # FastAPI serializa dict em JSON automaticamente
    return HTTPException(status_code=400 if code == "VALIDATION_ERROR" else 500, detail=payload["error"])


def score_to_level(p: float, model_name: str) -> RiskLevel:
    t = thresholds_for(model_name)
    if p >= t["high"]:
        return "high"
    if p >= t["low"]:
        return "medium"
    return "low"


def load_models() -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    for name, meta in MODEL_REGISTRY.items():
        path = Path(meta["pipeline_path"])
        if not path.exists():
            logger.error(f"Modelo não encontrado: {path}")
            raise RuntimeError(f"Modelo '{name}' não encontrado em {path}")
        loaded[name] = joblib.load(path)
        logger.info(f"Modelo carregado: {name} ({path})")
    return loaded


# Carrega na inicialização
try:
    MODELS = load_models()
except Exception as e:
    # Evita esconder falha de boot: preferível falhar cedo no deploy
    logger.exception("Falha ao carregar modelos na inicialização")
    raise


def get_models() -> Dict[str, Any]:
    return MODELS


def get_model(model_name: str, model_dependency: Dict[str, Any]) -> Any:
    if model_name not in model_dependency:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="NOT_FOUND",
                    message=f"Modelo '{model_name}' não encontrado",
                    details=[{"available_models": list(model_dependency.keys())}],
                )
            ).model_dump()["error"],
        )
    return model_dependency[model_name]


def predict_one(model: Any, model_name: str, features: Features) -> tuple[int, float, RiskLevel]:
    """Probabilidade calibrada + decisão pelo limiar de rastreio definido no treino."""
    df = features.to_frame()
    risk_score = max(0.0, min(1.0, float(model.predict_proba(df)[0][1])))
    prediction = int(risk_score >= thresholds_for(model_name)["low"])
    return prediction, risk_score, score_to_level(risk_score, model_name)


# Grupos de variáveis para a explicação (variáveis muito correlacionadas são explicadas em conjunto)
FACTOR_GROUPS: List[tuple[str, str, List[str]]] = [
    ("age", "Idade", ["age"]),
    ("sex", "Sexo", ["sex_male"]),
    ("education", "Escolaridade", ["education_years"]),
    ("body", "Medidas corporais (peso, altura, cintura)",
     ["height_cm", "weight_kg", "bmi", "waist_cm", "waist_height_ratio"]),
    ("bp", "Tensão arterial e hipertensão", ["sbp", "dbp", "bp_meds", "told_hypertension"]),
    ("tobacco", "Tabaco", ["smoker_current", "smokeless_current"]),
    ("alcohol", "Álcool", ["alcohol_past12m", "alcohol_days_month"]),
    ("diet", "Fruta e vegetais", ["fruit_servings_day", "veg_servings_day"]),
    ("activity", "Actividade física e sedentarismo", ["met_min_week", "sedentary_hours_day"]),
]
EXPLAIN_NOTE = (
    "Efeito aproximado de cada grupo: diferença entre o risco da pessoa e o risco que o modelo daria "
    "se esse grupo tivesse os valores típicos (mediana) da amostra de treino. Os efeitos não somam "
    "exactamente ao total e indicam associações estatísticas, não causas."
)
_REFERENCE_CACHE: Dict[str, pd.Series] = {}


def reference_row(model: Any, model_name: str) -> pd.Series:
    """Valores típicos (medianas de treino) usados pelo imputador do modelo."""
    if model_name not in _REFERENCE_CACHE:
        imputer = model.calibrated_classifiers_[0].estimator.named_steps["imputer"]
        _REFERENCE_CACHE[model_name] = pd.Series(imputer.statistics_, index=FEATURE_ORDER)
    return _REFERENCE_CACHE[model_name]


def explain_one(model: Any, model_name: str, features: Features, risk_score: float) -> Explanation:
    """
    Explicação por oclusão de grupos: para cada grupo, substitui os seus valores pelos de referência
    e mede quanto o risco muda. Funciona igual para os 4 modelos.
    """
    df = features.to_frame()
    ref = reference_row(model, model_name)
    rows = [ref.to_frame().T.astype(float)]
    for _, _, cols in FACTOR_GROUPS:
        alt = df.copy()
        alt[cols] = ref[cols].values
        rows.append(alt)
    probs = model.predict_proba(pd.concat(rows, ignore_index=True))[:, 1]
    baseline, without = float(probs[0]), probs[1:]
    factors = []
    for (key, label, cols), p_without in zip(FACTOR_GROUPS, without):
        provided = bool(df[cols].notna().any(axis=1).iloc[0])
        effect = float(risk_score - p_without) if provided else 0.0
        factors.append(FactorEffect(key=key, label=label, effect=effect, provided=provided, features=cols))
    factors.sort(key=lambda f: abs(f.effect), reverse=True)
    return Explanation(method="occlusion-by-group", baseline_score=baseline, factors=factors, note=EXPLAIN_NOTE)


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de Avaliação de Risco de Diabetes",
    version="1.0.0",
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Interface web do protótipo."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", time=now_iso())


@app.get(f"{API_PREFIX}/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    items: List[ModelSummary] = []
    for k, meta in MODEL_REGISTRY.items():
        items.append(
            ModelSummary(
                name=k,  # type: ignore[arg-type]
                model_id=meta["model_id"],
                model_version=meta["model_version"],
                has_predict_proba=hasattr(MODELS[k], "predict_proba"),
            )
        )
    return ModelsListResponse(available_models=items, total_models=len(items))


@app.get(f"{API_PREFIX}/models/{{model_name}}", response_model=ModelInfoResponse)
def model_info(model_name: ModelName) -> ModelInfoResponse:
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorInner(code="NOT_FOUND", message="Modelo não encontrado")
            ).model_dump()["error"],
        )
    meta = MODEL_REGISTRY[model_name]
    return ModelInfoResponse(
        name=model_name,
        model_id=meta["model_id"],
        model_version=meta["model_version"],
        thresholds=thresholds_for(model_name),
        glucose_threshold_mmol_l=META["glucose_threshold_mmol_l"],
        features=FEATURE_ORDER,
    )


@app.post(f"{API_PREFIX}/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model_dependency: Dict[str, Any] = Depends(get_models)) -> PredictResponse:
    model = get_model(req.model_name, model_dependency)

    request_id = str(uuid4())
    created_at = now_iso()

    try:
        pred, risk_score, risk_level = predict_one(model, req.model_name, req.features)
        explanation = explain_one(model, req.model_name, req.features, risk_score) if req.explain else None

        resp = PredictResponse(
            request_id=request_id,
            model=ModelRef(
                name=req.model_name,
                model_id=MODEL_REGISTRY[req.model_name]["model_id"],
                model_version=MODEL_REGISTRY[req.model_name]["model_version"],
            ),
            subject_id=req.subject_id,
            result=PredictResult(
                prediction=pred,  # type: ignore[arg-type]
                risk_score=risk_score,
                risk_level=risk_level,
            ),
            explanation=explanation,
            created_at=created_at,
        )

        logger.info(f"predict ok model={req.model_name} request_id={request_id} subject={req.subject_id}")
        return resp

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("validation error during prediction")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="VALIDATION_ERROR",
                    message="Erro na validação dos dados",
                    details=[{"exception": str(e)}],
                )
            ).model_dump()["error"],
        )
    except Exception as e:
        logger.exception("internal error during prediction")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="INTERNAL_ERROR",
                    message="Erro interno no servidor durante a predição",
                    details=[{"exception": str(e)}],
                )
            ).model_dump()["error"],
        )


@app.post(f"{API_PREFIX}/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest, model_dependency: Dict[str, Any] = Depends(get_models)
) -> BatchPredictResponse:
    model = get_model(req.model_name, model_dependency)

    batch_id = str(uuid4())
    created_at = now_iso()
    results: List[BatchPredictResult] = []

    try:
        for item in req.items:
            pred, risk_score, risk_level = predict_one(model, req.model_name, item.features)

            results.append(
                BatchPredictResult(
                    subject_id=item.subject_id,
                    prediction=pred,  # type: ignore[arg-type]
                    risk_score=risk_score,
                    risk_level=risk_level,
                )
            )

        logger.info(f"batch ok model={req.model_name} batch_id={batch_id} n={len(results)}")
        return BatchPredictResponse(
            batch_id=batch_id,
            model=ModelRef(
                name=req.model_name,
                model_id=MODEL_REGISTRY[req.model_name]["model_id"],
                model_version=MODEL_REGISTRY[req.model_name]["model_version"],
            ),
            results=results,
            created_at=created_at,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("validation error during batch prediction")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="VALIDATION_ERROR",
                    message="Erro na validação dos dados",
                    details=[{"exception": str(e)}],
                )
            ).model_dump()["error"],
        )
    except Exception as e:
        logger.exception("internal error during batch prediction")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="INTERNAL_ERROR",
                    message="Erro interno no servidor durante o batch scoring",
                    details=[{"exception": str(e)}],
                )
            ).model_dump()["error"],
        )


@app.get(f"{API_PREFIX}/metrics/{{model_name}}", response_model=MetricsResponse)
def get_model_metrics(model_name: ModelName) -> MetricsResponse:
    try:
        if not METRICS_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=ErrorInner(code="NOT_FOUND", message="Arquivo de métricas não encontrado")
                ).model_dump()["error"],
            )

        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

        model_name_lower = model_name.lower()
        for key in metrics.keys():
            if key.lower() == model_name_lower:
                return MetricsResponse(model=key, metrics=metrics[key])

        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="NOT_FOUND",
                    message=f"Modelo '{model_name}' não encontrado nas métricas",
                    details=[{"available_models": list(metrics.keys())}],
                )
            ).model_dump()["error"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("error reading metrics")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorInner(code="INTERNAL_ERROR", message="Erro ao carregar métricas", details=[{"exception": str(e)}])
            ).model_dump()["error"],
        )


@app.get(f"{API_PREFIX}/health/{{model_name}}", response_model=ModelHealthResponse)
def model_health(model_name: ModelName, model_dependency: Dict[str, Any] = Depends(get_models)) -> ModelHealthResponse:
    model = get_model(model_name, model_dependency)
    return ModelHealthResponse(
        model=model_name,
        status="healthy",
        has_predict_proba=hasattr(model, "predict_proba"),
        model_type=type(model).__name__,
    )