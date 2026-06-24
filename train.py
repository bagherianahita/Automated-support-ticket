"""Training: urgency + topic classifiers for support ticket triage."""

import json
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    return WS_RE.sub(" ", s).strip()


def combine_fields(subject: str, body: str, meta: dict | None = None) -> str:
    parts = [clean_text(subject), clean_text(body)]
    if meta:
        for key in ("product", "plan", "region"):
            if key in meta and meta[key]:
                parts.append(f"{key}:{str(meta[key]).lower()}")
    return " ".join(p for p in parts if p)


def find_best_threshold(
    pipe,
    X_valid: List[str],
    y_valid: List[str],
    positive_label: str = "high",
    target_precision: float = 0.80,
):
    proba = pipe.predict_proba(X_valid)
    clf = pipe.named_steps["clf"]
    classes = getattr(clf, "classes_", None)
    if classes is None and hasattr(clf, "calibrated_classifiers_"):
        classes = clf.calibrated_classifiers_[0].estimator.classes_
    idx_arr = np.where(classes == positive_label)[0]
    if len(idx_arr) == 0:
        return (0.5, 0.0, 0.0)
    idx = int(idx_arr[0])
    scores = proba[:, idx]
    best = (0.5, 0.0, 0.0)
    for thr in np.linspace(0.2, 0.9, 36):
        pred = np.where(scores >= thr, positive_label, "not_high")
        y_arr = np.array(y_valid)
        tp = np.sum((pred == positive_label) & (y_arr == positive_label))
        fp = np.sum((pred == positive_label) & (y_arr != positive_label))
        fn = np.sum((pred != positive_label) & (y_arr == positive_label))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        if precision >= target_precision and recall > best[2]:
            best = (thr, precision, recall)
    return best


def main():
    data_path = Path("data/tickets.csv")
    df = pd.read_csv(data_path)
    df["text"] = df.apply(
        lambda r: combine_fields(
            r.get("subject", ""),
            r.get("body", ""),
            {"product": r.get("product"), "plan": r.get("plan"), "region": r.get("region")},
        ),
        axis=1,
    )

    X = df["text"].tolist()
    y_urgency = df["urgency_label"].tolist()
    y_topic = df["topic_label"].tolist()

    stratify_u = y_urgency if df["urgency_label"].value_counts().min() >= 2 else None

    Xtr, Xte, yU_tr, yU_te, yT_tr, yT_te = train_test_split(
        X, y_urgency, y_topic, test_size=0.2, random_state=42, stratify=stratify_u
    )

    vectorizer = HashingVectorizer(
        n_features=2**18,
        ngram_range=(1, 2),
        norm="l2",
        alternate_sign=False,
        lowercase=False,
    )

    urg_base = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    cv_folds = min(3, min(df["urgency_label"].value_counts()))
    if cv_folds >= 2:
        urg_clf = Pipeline([("vec", vectorizer), ("clf", CalibratedClassifierCV(urg_base, cv=cv_folds))])
    else:
        urg_clf = Pipeline([("vec", vectorizer), ("clf", urg_base)])
    top_clf = Pipeline(
        [
            ("vec", vectorizer),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    urg_clf.fit(Xtr, yU_tr)
    top_clf.fit(Xtr, yT_tr)

    print("=== URGENCY ===")
    print(classification_report(yU_te, urg_clf.predict(Xte)))
    print("=== TOPIC ===")
    print(classification_report(yT_te, top_clf.predict(Xte)))

    thr, p, r = find_best_threshold(urg_clf, Xte, yU_te, positive_label="high", target_precision=0.85)
    if thr == 0.5 and p == 0.0:
        thr = 0.5
    print(f"High-urgency threshold: {thr:.2f} (precision={p:.2f}, recall={r:.2f})")

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    joblib.dump(urg_clf, artifacts / "urgency_clf.joblib")
    joblib.dump(top_clf, artifacts / "topic_clf.joblib")
    with open(artifacts / "urgency_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"high_threshold": thr}, f)


if __name__ == "__main__":
    main()
