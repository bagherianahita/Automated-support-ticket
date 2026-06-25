# Automated Support Ticket Triage

**ML-powered ticket routing** — classifies urgency and topic in under 200ms using lightweight scikit-learn models, designed for AWS Lambda deployment.
<img width="1211" height="903" alt="image" src="https://github.com/user-attachments/assets/662dae0f-911f-4ee0-b7eb-ff6daed17323" />
http://localhost:8010/
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
uvicorn app:app --port 8010 --host 127.0.0.1
```

Or:

```bash
python app.py
```

Open the **browser demo** (pre-filled ticket) or use Swagger **POST /triage**. Pre-trained models are in `artifacts/`.

| | URL |
|---|-----|
| **Web UI (demo)** | http://localhost:8010 |
| **API docs (Swagger)** | http://localhost:8010/docs |
| **Health check** | http://localhost:8010/health |

> Port **8010** avoids conflicts with other local APIs (e.g. MESO on 8000).

Example:

```bash
curl -X POST http://localhost:8010/triage \
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
