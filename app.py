"""FastAPI inference API for support ticket triage."""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from train import combine_fields

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8010

ARTIFACTS = Path("artifacts")


def _ensure_artifacts() -> None:
    if (ARTIFACTS / "urgency_clf.joblib").exists():
        return
    subprocess.check_call([sys.executable, "train.py"], cwd=Path(__file__).parent)


def _load_models():
    _ensure_artifacts()
    urgency = joblib.load(ARTIFACTS / "urgency_clf.joblib")
    topic = joblib.load(ARTIFACTS / "topic_clf.joblib")
    with open(ARTIFACTS / "urgency_threshold.json", encoding="utf-8") as f:
        thr = float(json.load(f).get("high_threshold", 0.5))
    return urgency, topic, thr


urgency_clf, topic_clf, HIGH_THR = _load_models()

app = FastAPI(
    title="Ticket Triage API",
    version="1.0.0",
    description="Support ticket triage demo — browser UI at / with defaults, or POST /triage in /docs.",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _urgency_classes(pipe):
    clf = pipe.named_steps["clf"]
    classes = getattr(clf, "classes_", None)
    if classes is None and hasattr(clf, "calibrated_classifiers_"):
        classes = clf.calibrated_classifiers_[0].estimator.classes_
    return classes


@app.get("/")
def demo_ui():
    demo_path = STATIC_DIR / "demo.html"
    if demo_path.exists():
        return FileResponse(demo_path)
    return {"message": "Ticket Triage API", "docs": "/docs", "health": "/health"}


class TicketIn(BaseModel):
    subject: str = Field(default="Cannot login", example="Cannot login")
    body: str = Field(default="I forgot my password and the reset link expired.", example="Reset link expired")
    product: Optional[str] = Field(default="saas", example="saas")
    plan: Optional[str] = Field(default="pro", example="pro")
    region: Optional[str] = Field(default="na", example="na")


class TriageOut(BaseModel):
    urgency: Literal["high", "normal", "low"]
    topic: str
    confidence: float
    route_queue: str
    latency_ms: float


def route_decision(urgency: str, topic: str) -> str:
    if urgency == "high":
        return f"{topic}_priority_queue"
    if topic == "billing":
        return "billing_queue"
    return f"{topic}_queue"


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/triage", response_model=TriageOut)
def triage(ticket: TicketIn):
    t0 = time.perf_counter()
    text = combine_fields(
        ticket.subject,
        ticket.body,
        {"product": ticket.product, "plan": ticket.plan, "region": ticket.region},
    )

    topic = topic_clf.predict([text])[0]
    urg_proba = urgency_clf.predict_proba([text])[0]
    urg_classes = _urgency_classes(urgency_clf)
    probs = dict(zip(urg_classes, urg_proba))
    urgency = max(probs, key=probs.get)
    if "high" in probs and probs["high"] >= HIGH_THR:
        urgency = "high"

    confidence = float(probs.get(urgency, max(probs.values())))
    latency = (time.perf_counter() - t0) * 1000.0

    return TriageOut(
        urgency=urgency,
        topic=topic,
        confidence=round(confidence, 4),
        route_queue=route_decision(urgency, topic),
        latency_ms=round(latency, 2),
    )


if __name__ == "__main__":
    import os

    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
