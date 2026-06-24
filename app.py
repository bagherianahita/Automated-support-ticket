"""FastAPI inference API for support ticket triage."""

import json
import time
from pathlib import Path
from typing import Literal, Optional

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from train import combine_fields

ARTIFACTS = Path("artifacts")
urgency_clf = joblib.load(ARTIFACTS / "urgency_clf.joblib")
topic_clf = joblib.load(ARTIFACTS / "topic_clf.joblib")
with open(ARTIFACTS / "urgency_threshold.json", encoding="utf-8") as f:
    HIGH_THR = float(json.load(f).get("high_threshold", 0.5))

app = FastAPI(title="Ticket Triage API", version="1.0.0")


class TicketIn(BaseModel):
    subject: str
    body: str
    product: Optional[str] = None
    plan: Optional[str] = None
    region: Optional[str] = None


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
    return {"status": "ok"}


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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
