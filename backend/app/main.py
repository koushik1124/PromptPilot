"""
PromptPilot Backend Application
"""

import os
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.prompt_analyzer import PromptAnalyzer
from app.services.prompt_enhancer import PromptEnhancer
from app.services.prompt_evaluator import PromptEvaluator

DEFAULT_DEV_ORIGINS: List[str] = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]


def parse_allowed_origins(raw: Optional[str]) -> List[str]:
    """
    Parse a comma-separated origin string into a clean list.
    Strips whitespace and excludes empty entries.
    """
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_allowed_origins() -> List[str]:
    """
    Determine allowed CORS origins based on the environment.
    - Production: requires ALLOWED_ORIGINS to be non-empty and forbids wildcard '*'.
    - Development: defaults to local development origins if ALLOWED_ORIGINS is not set.
    """
    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    raw = os.getenv("ALLOWED_ORIGINS", "")
    parsed = parse_allowed_origins(raw)

    if env == "production":
        if not parsed:
            raise RuntimeError(
                "ALLOWED_ORIGINS environment variable must be set in production mode."
            )
        if "*" in parsed:
            raise RuntimeError(
                "Wildcard '*' CORS origin is not permitted in production mode."
            )
        return parsed

    return parsed if parsed else list(DEFAULT_DEV_ORIGINS)


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    mode: Optional[str] = "auto"


class EvaluateRequest(BaseModel):
    original: str = Field(..., min_length=1, max_length=8000)
    enhanced: str = Field(..., min_length=1, max_length=8000)
    enhancement_overhead_tokens: Optional[int] = 0
    deep_verify: Optional[bool] = False


def create_app() -> FastAPI:
    application = FastAPI(title="PromptPilot API", version="1.0.0")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    analyzer = PromptAnalyzer()
    enhancer = PromptEnhancer()
    evaluator = PromptEvaluator()

    backend_api_key = os.getenv("BACKEND_API_KEY")

    def verify_api_key(x_api_key: Optional[str] = Header(None)):
        if backend_api_key and x_api_key != backend_api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @application.get("/health")
    def health_check():
        return {"status": "ok", "service": "PromptPilot Backend"}

    @application.post("/api/analyze", dependencies=[Depends(verify_api_key)])
    def analyze_prompt(payload: AnalyzeRequest):
        return analyzer.analyze(payload.prompt)

    @application.post("/api/enhance", dependencies=[Depends(verify_api_key)])
    def enhance_prompt(payload: EnhanceRequest):
        mode = payload.mode or "auto"
        if mode not in PromptEnhancer.ALL_ACCEPTED_MODES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mode. Valid modes: {sorted(PromptEnhancer.ALL_ACCEPTED_MODES)}",
            )

        result = enhancer.enhance(payload.prompt, mode=mode)
        if not result["success"]:
            raise HTTPException(status_code=502, detail=result["message"])
        return result

    @application.post("/api/evaluate", dependencies=[Depends(verify_api_key)])
    def evaluate_prompt(payload: EvaluateRequest):
        return evaluator.evaluate(
            original=payload.original,
            enhanced=payload.enhanced,
            enhancement_overhead_tokens=payload.enhancement_overhead_tokens or 0,
            deep_verify=payload.deep_verify or False,
        )

    return application


app = create_app()