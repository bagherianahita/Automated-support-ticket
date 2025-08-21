# Automated-support-ticket to reduce response time

training code, 

inference API (FastAPI), AWS Lambda adapter, and the tech decisions that hit two tough constraints: class imbalance and <200 ms latency.

Goal: Auto-triage support tickets by Urgency (high|normal|low) and Topic ( billing|bug|feature_request|account), 
then route to the right queue in real time.
Python for ML + API
scikit-learn (linear models) + HashingVectorizer (tiny memory, zero fit on vocab)
spaCy only for lightweight text cleanup (optional; avoid heavy models on Lambda)
FastAPI for a minimal REST API
Mangum to run FastAPI on AWS Lambda (API Gateway)
class_weight='balanced' + threshold tuning to handle class imbalance
Provisioned Concurrency (or warmed function) to hit p99 < 200 ms SLA
----------------
Why not large transformer models here?  (hugging face transformer library)
Cold start + model size would hurt latency and cost. 
A tuned linear model on hashed features is ~10–50× faster at inference and very stable.
------------------------
Data & features

Input: 
ticket text fields (subject, body), optional metadata (product, plan, region).

Preprocess: 
normalize case, strip URLs, collapse whitespace, keep a few signal tokens like !!!, error code, payment failed.

Vectorization: HashingVectorizer(ngram_range=(1,2), alternate_sign=False)

Pros: no fitted vocabulary (small artifact), ultra-fast, stable memory footprint.
HashingVectorizer(ngram_range=(1,2), alternate_sign=False)
-------------------------------------------------------------------------------------------
Handling class imbalance  
Model-side: class_weight='balanced' (re-weights minority class).
Probability calibration: CalibratedClassifierCV gives reliable probabilities.
Threshold tuning: Choose high-urgency threshold to meet business precision (≥0.85) while maximizing recall.
--------------------------------------------------------------------
Data-side (if available): Augment minority examples (paraphrases), or re-sample.
Monitoring: Track precision/recall per class in production; update threshold as class priors drift.
----------------------------------------------------------------------
Latency optimization checklist
Small, CPU-friendly models (linear, hashed features).
One vectorizer instance used by both models (pipeline-level).
Keep models loaded as globals (no per-request disk I/O).
JSON-only I/O, no heavy post-processing.
Lambda Provisioned Concurrency; short function memory burst (1024–2048 MB) for faster CPU.
Warm-up pings every 5 minutes (if can’t use provisioned concurrency.)
-----------------------------------------------------------------------------

Minimal requirements.txt

fastapi==0.111.0
uvicorn==0.30.1
mangum==0.17.0
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2


(Skip spaCy on Lambda unless   absolutely need it. If   do, use the smallest tokenizer-only setup and build a Lambda layer.)
-------------------------------------------------------

Optional: unit test (sanity)
# test_app.py
from app import combine_fields, clean_text

def test_clean():
    s = " Hello!!! Visit https://acme.com NOW  "
    out = clean_text(s)
    assert "http" not in out
    assert out == "hello!!! visit now"

def test_combine():
    t = combine_fields("PayMent Failed", "error code 402     please help", {"product":"Billing","plan":"Pro"})
    assert "product:billing" in t
    assert "plan:pro" in t
--------------------------------------------------------------------
used hashed n-grams + linear models to guarantee sub-200 ms latency on Lambda and keep artifacts tiny.
To address class imbalance, 
used balanced class weights + calibrated probabilities and tuned the high-urgency threshold
to meet business precision while improving recall of critical tickets.”
The FastAPI + Mangum adapter lets run the exact same code locally and on 
AWS Lambda/API Gateway with Provisioned Concurrency to avoid cold start spikes.”
log per-class precision/recall and percent routed per queue to spot drift and re-tune thresholds monthly.”
