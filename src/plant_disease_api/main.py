import logging
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from plant_disease_api.config import Settings, get_settings
from plant_disease_api.logging_config import configure_logging
from plant_disease_api.ml import PlantDiseasePredictor, Predictor
from plant_disease_api.rate_limit import InMemoryRateLimiter
from plant_disease_api.schemas import HealthResponse, PredictionResponse
from plant_disease_api.security import save_validated_upload


REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "path"])
PREDICTIONS = Counter("predictions_total", "Prediction requests", ["status"])
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    docs_url = "/docs" if settings.enable_docs else None
    redoc_url = "/redoc" if settings.enable_docs else None
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
    templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))
    static_dir = settings.base_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.predictor = predictor

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid4().hex)
        start = time.monotonic()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - start
            path = request.url.path
            REQUESTS.labels(request.method, path, str(status_code)).inc()
            LATENCY.labels(request.method, path).observe(elapsed)
            logger.info(
                "request_completed method=%s path=%s status=%s duration_seconds=%.4f",
                request.method,
                path,
                status_code,
                elapsed,
                extra={"request_id": request_id},
            )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    def get_predictor() -> Predictor:
        if app.state.predictor is None:
            app.state.predictor = PlantDiseasePredictor(settings)
        return app.state.predictor

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_exception")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health/live", response_model=dict[str, str], tags=["health"])
    async def liveness():
        return {"status": "ok"}

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def readiness(predictor: Predictor = Depends(get_predictor)):
        return HealthResponse(
            status="ok",
            model_loaded=True,
            classes=len(predictor.class_names),
            model_version=predictor.model_version,
        )

    @app.get("/metrics", tags=["monitoring"])
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", response_class=HTMLResponse, tags=["ui"])
    async def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": None,
                "confidence": None,
                "img_path": None,
                "top_predictions": [],
            },
        )

    @app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        explain: bool = Form(False),
        predictor: Predictor = Depends(get_predictor),
    ):
        limiter.check(request)
        saved_path = await save_validated_upload(
            file,
            upload_dir=settings.upload_dir,
            allowed_extensions=settings.allowed_extensions,
            max_upload_mb=settings.max_upload_mb,
        )

        try:
            result = predictor.predict(saved_path, explain=explain)
            PREDICTIONS.labels("success").inc()
            return result
        except Exception:
            PREDICTIONS.labels("error").inc()
            saved_path.unlink(missing_ok=True)
            logger.exception("prediction_failed")
            raise

    @app.post("/", response_class=HTMLResponse, tags=["ui"])
    async def predict_form(
        request: Request,
        file: UploadFile = File(...),
        predictor: Predictor = Depends(get_predictor),
    ):
        try:
            saved_path = await save_validated_upload(
                file,
                upload_dir=settings.upload_dir,
                allowed_extensions=settings.allowed_extensions,
                max_upload_mb=settings.max_upload_mb,
            )
            result = predictor.predict(saved_path, explain=True)
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "img_path": result.image_url,
                    "top_predictions": result.top_predictions,
                    "gradcam_path": result.gradcam_url,
                    "uncertain": result.uncertain,
                },
            )
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": getattr(exc, "detail", "Prediction failed."),
                    "prediction": None,
                    "confidence": None,
                    "img_path": None,
                    "top_predictions": [],
                },
                status_code=getattr(exc, "status_code", 500),
            )

    @app.get("/api/v1/classes", response_model=list[str], tags=["metadata"])
    async def classes(predictor: Predictor = Depends(get_predictor)):
        return predictor.class_names

    return app


app = create_app()
