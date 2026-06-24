# Automated Support Ticket Triage

**ML-powered ticket routing** — classifies urgency and topic in under 200ms using lightweight scikit-learn models, designed for AWS Lambda deployment.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square)

---

## Architecture

```
┌─────────────┐   POST /triage   ┌────────────────────────────────────┐
│ Ticket API  │ ───────────────► │  FastAPI (app.py)                  │
│ or Lambda   │ ◄─────────────── │  HashingVectorizer + SGD / LogReg  │
└─────────────┘   urgency+topic  └────────────────────────────────────┘
                                              ▲
                                   train.py → artifacts/*.joblib
```

---

## Quick start

```bash
pip install -r requirements.txt
python train.py              # trains on data/tickets.csv
uvicorn app:app --reload     # http://localhost:8000/docs
```

Example:

```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d "{\"subject\":\"Cannot login\",\"body\":\"Reset link expired\"}"
```

---

## Project structure

| File | Purpose |
|------|---------|
| `train.py` | Train urgency + topic classifiers |
| `app.py` | FastAPI inference API |
| `data/tickets.csv` | Sample training data |
| `artifacts/` | Saved models (generated) |

---

## License

MIT — see [LICENSE](LICENSE).
