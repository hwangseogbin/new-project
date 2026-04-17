from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from flask import Flask, redirect, render_template, request, session, url_for

from article_fetcher import fetch_article
from model import FakeNewsDetector


BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_DATASET_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl"}
DEFAULT_DATASET_LOCATIONS = (
    BASE_DIR / "data" / "WELFake_Dataset.csv",
)


def _resolve_dataset_candidate(candidate: Path) -> Path:
    if candidate.exists() and candidate.is_dir():
        supported_files = sorted(
            path
            for path in candidate.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
        )
        if supported_files:
            preferred_name = candidate.name.casefold()
            preferred_match = next(
                (
                    path
                    for path in supported_files
                    if path.stem.casefold() == preferred_name
                    or path.name.casefold().startswith(preferred_name)
                ),
                None,
            )
            return (preferred_match or supported_files[0]).resolve()
    return candidate.resolve()


def _resolve_dataset_path() -> Path:
    configured_path = os.getenv("DATASET_PATH", "").strip()
    if configured_path:
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        return _resolve_dataset_candidate(candidate)

    for default_candidate in DEFAULT_DATASET_LOCATIONS:
        resolved_candidate = _resolve_dataset_candidate(default_candidate)
        if resolved_candidate.exists():
            return resolved_candidate

    return _resolve_dataset_candidate(DEFAULT_DATASET_LOCATIONS[0])


def _normalize_fallback_url(raw_url: str) -> str:
    cleaned = str(raw_url or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.split()[0]
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"
    return cleaned


def _derive_link_topic(url: str) -> str:
    parsed = urlparse(url)
    segments = [
        unquote(segment).strip()
        for segment in parsed.path.split("/")
        if segment.strip()
    ]
    ignored_segments = {"news", "article", "articles", "story", "stories", "amp", "index", "index.html"}
    topic_seed = next(
        (segment for segment in reversed(segments) if segment.casefold() not in ignored_segments),
        "",
    )
    if not topic_seed:
        topic_seed = parsed.netloc or parsed.path or url

    topic = re.sub(r"[-_+/]+", " ", topic_seed)
    topic = re.sub(r"\.[a-z0-9]{2,5}$", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+", " ", topic).strip(" -_/")
    if not topic:
        return "Unknown Link"
    return topic[:120].title()


def _build_url_fallback_result(raw_url: str, reason: str) -> dict[str, object]:
    normalized_url = _normalize_fallback_url(raw_url)
    parsed = urlparse(normalized_url)
    domain = (parsed.hostname or parsed.netloc or "").lower().removeprefix("www.")
    source = domain or "Unknown source"
    title = _derive_link_topic(normalized_url or raw_url)
    fallback_text = (
        "This link could not be read as a full text article, so VerifiJin is using only the "
        "domain, link wording, and limited source cues for a basic check. Treat this result as "
        "low confidence, open the original article page if available, and compare the claim with "
        "trusted reporting before deciding whether it is real or fake."
    )
    recommendation = (
        "We could not read article text from this link, so this is a limited URL-based check. "
        "Open the original article page or paste the article text for a stronger verdict."
    )
    preview_reason = reason.strip() or "Readable article text was not available for this link."
    source_profile = detector._build_source_profile(url=normalized_url, source=source, author=source)
    signal_analysis = detector._analyze_signals(
        title=title,
        author=source,
        text=fallback_text,
        source_profile=source_profile,
        published_at="",
    )
    metadata_cues = ["URL-only fallback used", "Readable article text unavailable", *signal_analysis["metadata_cues"]]
    result = {
        "label": "Needs More Context",
        "label_code": -1,
        "confidence": 50.0,
        "probabilities": {"real": 50.0, "fake": 50.0},
        "input_quality": {
            "is_weak": True,
            "characters": len(" ".join(part for part in [source, title, fallback_text] if part)),
            "words": len(" ".join(part for part in [source, title, fallback_text] if part).split()),
            "recommendation": recommendation,
        },
        "analysis": {
            "source_cues": signal_analysis["source_cues"],
            "reporting_cues": [],
            "risk_cues": [],
            "metadata_cues": metadata_cues,
            "model_consensus": "url-only fallback",
            "secondary_model": "Unavailable",
        },
        "source_profile": {
            "label": source_profile["label"],
            "domain": source_profile["domain"],
            "score": round(source_profile["raw_score"] * 100, 2),
            "tier": source_profile["tier"],
            "kind": source_profile["kind"],
            "reason": source_profile["reason"],
        },
        "verification_links": detector._build_verification_links(
            title=title,
            source_profile=source_profile,
            url=normalized_url,
        ),
        "model_breakdown": {
            "headline_model": {"real": 50.0, "fake": 50.0},
            "body_model": {"real": 50.0, "fake": 50.0},
            "fusion_model": {"real": 50.0, "fake": 50.0},
            "secondary_model": None,
        },
        "article": {
            "url": normalized_url,
            "title": title,
            "author": source,
            "source": source,
            "domain": domain,
            "published_at": "",
            "preview": preview_reason,
            "input_mode": "url",
        },
    }
    detector._save_prediction(
        title=title,
        author=source,
        text=fallback_text,
        label=result["label"],
        confidence=result["confidence"],
        source=result["source_profile"]["label"],
        source_domain=result["source_profile"]["domain"],
        published_at="",
        url=normalized_url,
        input_mode="url",
        source_score=result["source_profile"]["score"],
        consensus="url-only fallback",
        recommendation=recommendation,
    )
    return result


app = Flask(__name__)
app.secret_key = "veritasai-secret-key"
detector = FakeNewsDetector(
    dataset_path=None,
    artifact_dir=BASE_DIR / "artifacts",
)
ADMIN_USERNAME = "hsb"
ADMIN_PASSWORD = "hsb18"


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    error = None
    if request.method == "POST":
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", "")).strip()
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.get("/admin")
def admin_dashboard() -> str:
    if not session.get("is_admin"):
        return redirect(url_for("login"))
    return render_template(
        "admin.html",
        metrics=detector.safe_metrics(),
        history=detector.get_prediction_history(limit=20),
    )


@app.post("/logout")
def logout() -> str:
    session.clear()
    return redirect(url_for("index"))


@app.get("/api/health")
def health() -> tuple[dict[str, object], int]:
    try:
        metrics = detector.ensure_ready()
    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}, 503
    return {
        "status": "ok",
        "model_ready": True,
        "metrics": metrics,
    }, 200


@app.get("/api/metrics")
def metrics() -> tuple[dict[str, object], int]:
    try:
        return detector.ensure_ready(), 200
    except FileNotFoundError as exc:
        return {"error": str(exc)}, 503


@app.post("/api/predict")
def predict() -> tuple[dict[str, object], int]:
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    author = str(payload.get("author") or payload.get("source") or "").strip()
    text = str(payload.get("text", "")).strip()
    url = str(payload.get("url", "")).strip()
    published_at = str(payload.get("published_at", "")).strip()

    if not any([title, author, text]):
        return {"error": "Provide at least one of title, author, or text."}, 400

    try:
        result = detector.predict(
            title=title,
            author=author,
            text=text,
            url=url,
            source=author,
            published_at=published_at,
        )
        preview_words = text.split()
        result["article"] = {
            "url": url,
            "title": title or "Manual article input",
            "author": author,
            "source": author or "Manual source",
            "domain": (urlparse(url).hostname or "").lower().removeprefix("www.") if url else "",
            "published_at": published_at,
            "preview": " ".join(preview_words[:45]) + ("..." if len(preview_words) > 45 else ""),
            "input_mode": "manual",
        }
        return result, 200
    except FileNotFoundError as exc:
        return {"error": str(exc)}, 503


@app.post("/api/predict-url")
def predict_url() -> tuple[dict[str, object], int]:
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()

    if not url:
        return {"error": "Provide a news article URL."}, 400

    try:
        article = fetch_article(url)
        result = detector.predict(
            title=article.title,
            author=article.author,
            text=article.text,
            url=article.url,
            source=article.source,
            published_at=article.published_at,
        )
        preview_words = article.text.split()
        result["article"] = {
            "url": article.url,
            "title": article.title,
            "author": article.author,
            "source": article.source,
            "domain": article.domain,
            "published_at": article.published_at,
            "preview": " ".join(preview_words[:45]) + ("..." if len(preview_words) > 45 else ""),
            "input_mode": "url",
        }
        return result, 200
    except FileNotFoundError as exc:
        return {"error": str(exc)}, 503
    except (ValueError, requests.RequestException) as exc:
        return _build_url_fallback_result(url, str(exc)), 200


@app.get("/api/history")
def history() -> tuple[dict[str, object], int]:
    limit = request.args.get("limit", default=8, type=int) or 8
    limit = max(1, min(limit, 50))
    return {"items": detector.get_prediction_history(limit=limit)}, 200


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )
