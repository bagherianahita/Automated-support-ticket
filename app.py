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
from pydantic import BaseModel, Field

from train import combine_fields

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
    description="Try POST /triage with the default example in /docs — no setup required.",
)


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
    urg_classes = urgency_clf.named_steps["clf"].classes_
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
    uvicorn.run("app:app", host="0.0.0.0", port=int(__import__("os").getenv("PORT", "8010")), reload=True)
