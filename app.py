"""
API de Avaliação de Risco de Diabetes (BRFSS) - FastAPI

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

# Registry de modelos (inclui metadata para rastreabilidade)
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "rf": {
        "pipeline_path": str(MODELS_DIR / "pipeline_rf.pkl"),
        "model_id": "diabetes-risk-rf",
        "model_version": "1.0.0",
    },
    "lgbm": {
        "pipeline_path": str(MODELS_DIR / "pipeline_lgbm.pkl"),
        "model_id": "diabetes-risk-lgbm",
        "model_version": "1.0.0",
    },
    "xgb": {
        "pipeline_path": str(MODELS_DIR / "pipeline_xgb.pkl"),
        "model_id": "diabetes-risk-xgb",
        "model_version": "1.0.0",
    },
}

# Thresholds para estratificação (ajuste conforme tua calibração/estratégia)
THRESHOLDS = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
}

ModelName = Literal["rf", "lgbm", "xgb"]
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
    Features BRFSS diabetes indicators.
    """

    # binárias 0/1
    HighBP: conint(ge=0, le=1)
    HighChol: conint(ge=0, le=1)
    CholCheck: conint(ge=0, le=1)
    Smoker: conint(ge=0, le=1)
    Stroke: conint(ge=0, le=1)
    HeartDiseaseorAttack: conint(ge=0, le=1)
    PhysActivity: conint(ge=0, le=1)
    Fruits: conint(ge=0, le=1)
    Veggies: conint(ge=0, le=1)
    HvyAlcoholConsump: conint(ge=0, le=1)
    AnyHealthcare: conint(ge=0, le=1)
    NoDocbcCost: conint(ge=0, le=1)
    DiffWalk: conint(ge=0, le=1)
    Sex: conint(ge=0, le=1)

    # contínuas/ordinais
    BMI: confloat(ge=10, le=80)         # range plausível
    GenHlth: conint(ge=1, le=5)         # 1..5
    MentHlth: conint(ge=0, le=30)       # dias
    PhysHlth: conint(ge=0, le=30)       # dias

    # categóricas ordinais
    Age: conint(ge=1, le=13)            # _AGEG5YR
    Education: conint(ge=1, le=6)       # BRFSS comum
    Income: conint(ge=1, le=8)          # BRFSS comum


class PredictRequest(BaseModel):
    model_name: ModelName = Field(..., description="Modelo a usar: rf|lgbm|xgb")
    subject_id: Optional[str] = Field(None, description="ID do utente/participante (opcional)")
    features: Features
    explain: bool = Field(False, description="Reservado para futuro (explicabilidade)")
    persist: bool = Field(False, description="Reservado para futuro (auditoria/armazenamento)")


class ModelRef(BaseModel):
    name: ModelName
    model_id: str
    model_version: str


class PredictResult(BaseModel):
    prediction: Literal[0, 1]
    risk_score: confloat(ge=0, le=1)
    risk_level: RiskLevel


class PredictResponse(BaseModel):
    request_id: str
    model: ModelRef
    subject_id: Optional[str]
    result: PredictResult
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


def score_to_level(p: float) -> RiskLevel:
    if p >= THRESHOLDS["high"]:
        return "high"
    if p >= THRESHOLDS["medium"]:
        return "medium"
    return "low"


def predict_proba_safe(model: Any, df: pd.DataFrame) -> Optional[float]:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(df)
    return float(proba[0][1])


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


def ensure_feature_order(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return df

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorInner(
                    code="VALIDATION_ERROR",
                    message="Features incompatíveis com o modelo",
                    details=[{"missing": missing, "extra": extra}],
                )
            ).model_dump()["error"],
        )
    return df[list(expected)]


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
        thresholds=THRESHOLDS,
    )


@app.post(f"{API_PREFIX}/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model_dependency: Dict[str, Any] = Depends(get_models)) -> PredictResponse:
    model = get_model(req.model_name, model_dependency)

    request_id = str(uuid4())
    created_at = now_iso()

    try:
        # DataFrame + compatibilidade de dtype com treino
        df = pd.DataFrame([req.features.model_dump()]).astype(float)
        df = ensure_feature_order(df, model)

        pred = model.predict(df)
        proba = predict_proba_safe(model, df)

        risk_score = float(proba) if proba is not None else float(pred[0])
        risk_score = max(0.0, min(1.0, risk_score))  # clamp defensivo
        risk_level = score_to_level(risk_score)

        resp = PredictResponse(
            request_id=request_id,
            model=ModelRef(
                name=req.model_name,
                model_id=MODEL_REGISTRY[req.model_name]["model_id"],
                model_version=MODEL_REGISTRY[req.model_name]["model_version"],
            ),
            subject_id=req.subject_id,
            result=PredictResult(
                prediction=int(pred[0]),  # type: ignore[arg-type]
                risk_score=risk_score,
                risk_level=risk_level,
            ),
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
            df = pd.DataFrame([item.features.model_dump()]).astype(float)
            df = ensure_feature_order(df, model)

            pred = model.predict(df)
            proba = predict_proba_safe(model, df)

            risk_score = float(proba) if proba is not None else float(pred[0])
            risk_score = max(0.0, min(1.0, risk_score))
            risk_level = score_to_level(risk_score)

            results.append(
                BatchPredictResult(
                    subject_id=item.subject_id,
                    prediction=int(pred[0]),  # type: ignore[arg-type]
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