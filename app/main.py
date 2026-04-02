import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.orchestrator import SupportOrchestrator
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    orchestrator = SupportOrchestrator()
    app.state.preloaded_files = orchestrator.preload()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(health_router)
app.include_router(query_router, prefix="/api")

if importlib.util.find_spec("multipart") is not None:
    from app.api.routes.upload import router as upload_router

    app.include_router(upload_router, prefix="/api")

static_dir = Path(__file__).resolve().parent / "static"

if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "message": "E-Commerce Support API is running.",
                "frontend": "Run the separate React frontend at http://localhost:5173",
                "health": "/health",
                "upload": "/api/upload",
                "query": "/api/query",
            }
        )
