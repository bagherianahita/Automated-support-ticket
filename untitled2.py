
"""1) Training code (two classifiers: Urgency + Topic)"""

# train.py
import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# -----------------------
# 1) Minimal text cleaner
# -----------------------
URL_RE = re.compile(r'https?://\S+')
WS_RE  = re.compile(r'\s+')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub(' ', s)
    s = s.replace('\n', ' ')
    s = WS_RE.sub(' ', s).strip()
    return s

def combine_fields(subject: str, body: str, meta: dict = None) -> str:
    parts = [clean_text(subject), clean_text(body)]
    if meta:
        # Optional light features from metadata
        for k in ('product','plan','region'):
            if k in meta and meta[k]:
                parts.append(f"{k}:{str(meta[k]).lower()}")
    return " ".join(p for p in parts if p)


import pandas as pd
from datasets import load_dataset

# Load the dataset
# df = load_dataset("interneuronai/companyx_customer_support_ticket_routing_distilbert_dataset")
df = pd.read_csv("tickets.csv")

train_df = df['train'].to_pandas()

# Display the first 6 rows
print(train_df.head(6))

df['text'] = df.apply(lambda r: combine_fields(
    r.get('subject',''), r.get('body',''),
    {'product': r.get('product'), 'plan': r.get('plan'), 'region': r.get('region')}
), axis=1)

# -----------------------
# 3) Split data
# -----------------------
X = df['text'].values
y_urgency = df['urgency_label'].values      # e.g., high|normal|low
y_topic   = df['topic_label'].values        # e.g., billing|bug|feature_request|account

Xtr, Xte, yU_tr, yU_te = train_test_split(X, y_urgency, test_size=0.2, random_state=42, stratify=y_urgency)
_,   _,   yT_tr, yT_te = train_test_split(X, y_topic,   test_size=0.2, random_state=42, stratify=y_topic)

# -----------------------
# 4) Vectorizer (fast & small)
# -----------------------
vectorizer = HashingVectorizer(
    n_features=2**20,          # tune vs. memory
    ngram_range=(1,2),
    norm='l2',
    alternate_sign=False,      # improves Logistic/SGD stability
    lowercase=False            # we already lowercased
)

# -----------------------
# 5) Models
# -----------------------
# Urgency: imbalanced => linear SVM (SGD hinge/log) with class_weight
urg_base = SGDClassifier(
    loss='log_loss',           # probabilistic; good for calibration
    class_weight='balanced',
    alpha=1e-5,               #overfitting
    max_iter=2000,
    tol=1e-3,               #stop learning
    n_jobs=-1,                 # all cpu
    random_state=42
)
urg_clf = Pipeline([
    ('vec', vectorizer),
    ('clf', CalibratedClassifierCV(urg_base, cv=3))  # better probs for thresholding
])

# Topic: usually multi-class, often more balanced
top_base = LogisticRegression(
    class_weight=None,
    max_iter=1000,
    n_jobs=-1
)
top_clf = Pipeline([
    ('vec', vectorizer),
    ('clf', top_base)
])

# -----------------------
# 6) Train
# -----------------------
urg_clf.fit(Xtr, yU_tr)
top_clf.fit(Xtr, yT_tr)

# -----------------------
# 7) Evaluate
# -----------------------
print("=== URGENCY ===")
yU_pred = urg_clf.predict(Xte)
print(classification_report(yU_te, yU_pred))

print("=== TOPIC ===")
yT_pred = top_clf.predict(Xte)
print(classification_report(yT_te, yT_pred))

# Optional: tune a custom threshold for "high" urgency to boost recall at fixed precision
def find_best_threshold(pipe, X_valid: List[str], y_valid: List[str], positive_label='high', target_precision=0.80):
    proba = pipe.predict_proba(X_valid)
    # proba[:, idx_of_high]
    classes = pipe.named_steps['clf'].classes_
    idx = int(np.where(classes == positive_label)[0])
    scores = proba[:, idx]
    # sweep thresholds
    best = (0.5, 0.0, 0.0)  # (thr, precision, recall)
    for thr in np.linspace(0.2, 0.9, 36):
        pred = np.where(scores >= thr, positive_label, 'not_high')
        # compute precision/recall
        tp = np.sum((pred == positive_label) & (np.array(y_valid) == positive_label))
        fp = np.sum((pred == positive_label) & (np.array(y_valid) != positive_label))
        fn = np.sum((pred != positive_label) & (np.array(y_valid) == positive_label))
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        if precision >= target_precision and recall > best[2]:
            best = (thr, precision, recall)
    return best

thr, p, r = find_best_threshold(urg_clf, Xte, yU_te, positive_label='high', target_precision=0.85)
print(f"Chosen high-urgency threshold: {thr:.2f} (precision={p:.2f}, recall={r:.2f})")

# -----------------------
# 8) Persist
# -----------------------
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(urg_clf, "artifacts/urgency_clf.joblib")
joblib.dump(top_clf, "artifacts/topic_clf.joblib")
with open("artifacts/urgency_threshold.json", "w") as f:
    json.dump({"high_threshold": thr}, f)








"""2) Real-time inference API (FastAPI)"""  using trained ml

# app.py
import json
import joblib
import time
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal

from train import clean_text, combine_fields  # reuse the exact same cleaning

ARTIFACTS = Path("artifacts") #folder
urgency_clf = joblib.load(ARTIFACTS / "urgency_clf.joblib")
topic_clf   = joblib.load(ARTIFACTS / "topic_clf.joblib")  # file
with open(ARTIFACTS / "urgency_threshold.json") as f:
    URG_CONF = json.load(f)
HIGH_THR = float(URG_CONF.get("high_threshold", 0.5)) 

app = FastAPI(title="Ticket Triage API", version="1.0.0")

class TicketIn(BaseModel): #inpyt
    subject: str
    body: str
    product: Optional[str] = None
    plan: Optional[str] = None
    region: Optional[str] = None

class TriageOut(BaseModel):   #output
    urgency: Literal["high","normal","low"]
    topic: str
    confidence: float
    route_queue: str
    latency_ms: float

def route_decision(urgency: str, topic: str) -> str:
    # Simple rules; make this a config in production
    if urgency == "high":
        return f"{topic}_priority_queue"
    if topic in {"billing"}:
        return "billing_queue"
    return f"{topic}_queue"

@app.post("/triage", response_model=TriageOut)
def triage(ticket: TicketIn):
    t0 = time.perf_counter()

    text = combine_fields(ticket.subject, ticket.body, {
        "product": ticket.product,
        "plan": ticket.plan,
        "region": ticket.region
    })

    # Predict topic (argmax)
    topic = topic_clf.predict([text])[0]
    # Predict urgency via calibrated probabilities + threshold for "high"
    urg_proba = urgency_clf.predict_proba([text])[0]
    urg_classes = urgency_clf.named_steps['clf'].classes_
    probs = dict(zip(urg_classes, urg_proba))
    # choose label with max probability
    urgency = max(probs, key=probs.get)

    # Apply custom rule: if "high" prob >= HIGH_THR -> force high
    if 'high' in probs and probs['high'] >= HIGH_THR:
        urgency = 'high'

    # Confidence (of chosen label)
    confidence = float(probs.get(urgency, max(probs.values())))

    route = route_decision(urgency, topic)
    latency = (time.perf_counter() - t0) * 1000.0

    return TriageOut(
        urgency=urgency,
        topic=topic,
        confidence=round(confidence, 4),
        route_queue=route,
        latency_ms=round(latency, 2)
    )

if __name__ == "__main__":
    # local dev
    uvicorn.run("app:triage", host="0.0.0.0", port=8000, reload=True)
