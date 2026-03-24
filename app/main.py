import json
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, select, func

from starlette.requests import Request

from app.detector import PositionalDetector
from app.models import RequestLog, create_db, engine, log_request

load_dotenv()

ANTHROPIC_API_URL = os.getenv("TARGET_API_URL", "https://api.anthropic.com/v1/messages")
FORWARDED_HEADERS = {"x-api-key", "anthropic-version", "content-type", "anthropic-beta"}

detector: PositionalDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    print("[PromptShield] Creating database...")
    create_db()
    print("[PromptShield] Loading detector model and building FAISS index...")
    detector = PositionalDetector()
    print("[PromptShield] Detector ready.")
    yield


app = FastAPI(title="PromptShield", lifespan=lifespan)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Proxy ─────────────────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    start = time.monotonic()
    body = await request.json()
    messages = body.get("messages", [])
    request_preview = json.dumps(body)[:200]

    result = detector.scan(messages)

    if result.is_suspicious:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_request(
            flagged=True,
            score=result.score,
            flagged_content=result.flagged_content[:200],
            reason=result.reason,
            detection_method=result.detection_method,
            request_preview=request_preview,
            latency_ms=latency_ms,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "flagged": True,
                "score": result.score,
                "reason": result.reason,
                "detection_method": result.detection_method,
            },
        )

    # Forward to Anthropic
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() in FORWARDED_HEADERS
    }
    if "x-api-key" not in forward_headers:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            forward_headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=60.0) as client:
        upstream = await client.post(ANTHROPIC_API_URL, json=body, headers=forward_headers)

    latency_ms = int((time.monotonic() - start) * 1000)
    log_request(
        flagged=False,
        score=result.score,
        flagged_content="",
        reason="",
        detection_method="",
        request_preview=request_preview,
        latency_ms=latency_ms,
    )

    return JSONResponse(status_code=upstream.status_code, content=upstream.json())


# ── Dashboard API ──────────────────────────────────────────────────────────────

@app.get("/api/logs")
def get_logs(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    offset = (page - 1) * page_size
    with Session(engine) as session:
        total = session.exec(select(func.count()).select_from(RequestLog)).one()
        logs = session.exec(
            select(RequestLog).order_by(RequestLog.id.desc()).offset(offset).limit(page_size)
        ).all()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [log.model_dump() for log in logs],
    }


@app.get("/api/logs/{log_id}")
def get_log(log_id: int):
    with Session(engine) as session:
        log = session.get(RequestLog, log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log entry not found")
    return log.model_dump()


@app.get("/api/stats")
def get_stats():
    with Session(engine) as session:
        total = session.exec(select(func.count()).select_from(RequestLog)).one()
        flagged = session.exec(
            select(func.count()).select_from(RequestLog).where(RequestLog.flagged == True)
        ).one()
        avg_latency = session.exec(
            select(func.avg(RequestLog.latency_ms)).select_from(RequestLog)
        ).one()

    flag_rate = round((flagged / total * 100), 2) if total > 0 else 0.0
    return {
        "total_requests": total,
        "total_flagged": flagged,
        "flag_rate_pct": flag_rate,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0.0,
    }
