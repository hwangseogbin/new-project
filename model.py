from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from math import log1p
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _load_stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            return set(FALLBACK_STOP_WORDS)


@dataclass
class TrainingArtifacts:
    headline_pipeline: Any
    body_pipeline: Any
    fusion_classifier: Any
    metrics: dict[str, Any]


class _HeuristicTextPipeline:
    def __init__(self, detector: "FakeNewsDetector", mode: str) -> None:
        self.detector = detector
        self.mode = mode

    def predict_proba(self, samples: Any) -> list[list[float]]:
        return [
            list(self.detector._heuristic_text_probabilities(str(sample), mode=self.mode))
            for sample in samples
        ]


class _HeuristicFusionClassifier:
    def __init__(self, detector: "FakeNewsDetector") -> None:
        self.detector = detector

    def predict_proba(self, rows: Any) -> list[list[float]]:
        return [
            list(self.detector._heuristic_fusion_probabilities(list(row)))
            for row in rows
        ]


class FakeNewsDetector:
    MODEL_VERSION = "4.1"
    TRANSFORMER_REPO_DIR = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--mrm8488--bert-tiny-finetuned-fake-news-detection"
    )
    SOURCE_PROFILE_RULES = (
        ("reuters.com", "Reuters", 0.96, "newsroom"),
        ("reuters", "Reuters", 0.96, "newsroom"),
        ("apnews.com", "Associated Press", 0.95, "newsroom"),
        ("associated press", "Associated Press", 0.95, "newsroom"),
        ("nytimes.com", "The New York Times", 0.92, "newsroom"),
        ("the new york times", "The New York Times", 0.92, "newsroom"),
        ("washingtonpost.com", "The Washington Post", 0.90, "newsroom"),
        ("washington post", "The Washington Post", 0.90, "newsroom"),
        ("bbc.com", "BBC", 0.92, "newsroom"),
        ("bbc.co.uk", "BBC", 0.92, "newsroom"),
        ("bbc", "BBC", 0.92, "newsroom"),
        ("npr.org", "NPR", 0.90, "newsroom"),
        ("npr", "NPR", 0.90, "newsroom"),
        ("bloomberg.com", "Bloomberg", 0.90, "newsroom"),
        ("bloomberg", "Bloomberg", 0.90, "newsroom"),
        ("theguardian.com", "The Guardian", 0.88, "newsroom"),
        ("the guardian", "The Guardian", 0.88, "newsroom"),
        ("ft.com", "Financial Times", 0.90, "newsroom"),
        ("financial times", "Financial Times", 0.90, "newsroom"),
        ("wsj.com", "Wall Street Journal", 0.89, "newsroom"),
        ("wall street journal", "Wall Street Journal", 0.89, "newsroom"),
        ("cnn.com", "CNN", 0.74, "newsroom"),
        ("cnn", "CNN", 0.74, "newsroom"),
        ("foxnews.com", "Fox News", 0.56, "newsroom"),
        ("fox news", "Fox News", 0.56, "newsroom"),
        ("newsmax.com", "Newsmax", 0.42, "partisan"),
        ("breitbart.com", "Breitbart", 0.34, "partisan"),
        ("facebook.com", "Facebook", 0.24, "platform"),
        ("twitter.com", "X / Twitter", 0.26, "platform"),
        ("x.com", "X / Twitter", 0.26, "platform"),
        ("youtube.com", "YouTube", 0.31, "platform"),
        ("tiktok.com", "TikTok", 0.24, "platform"),
        ("instagram.com", "Instagram", 0.24, "platform"),
        ("blogspot.com", "Blogspot", 0.28, "platform"),
        ("medium.com", "Medium", 0.40, "platform"),
    )
    SOURCE_CUES = {
        "reuters": "Reuters source mentioned",
        "associated press": "Associated Press source mentioned",
        "the new york times": "The New York Times source mentioned",
        "new york times": "The New York Times source mentioned",
        "washington post": "The Washington Post source mentioned",
        "bbc": "BBC source mentioned",
        "cnn": "CNN source mentioned",
        "npr": "NPR source mentioned",
        "bloomberg": "Bloomberg source mentioned",
        "the guardian": "The Guardian source mentioned",
        "financial times": "Financial Times source mentioned",
        "wall street journal": "Wall Street Journal source mentioned",
    }
    REPORTING_CUES = {
        "officials": "Official statement language detected",
        "statement": "Statement language detected",
        "according to": "Attribution language detected",
        "published": "Published-report wording detected",
        "report": "Report wording detected",
        "ministry": "Government reporting cue detected",
        "department": "Institutional reporting cue detected",
        "agency": "Agency reporting cue detected",
        "researchers": "Research reporting cue detected",
        "study": "Study/report wording detected",
        "police": "Institutional reporting cue detected",
        "court": "Court reporting cue detected",
        "parliament": "Government reporting cue detected",
        "spokesperson": "Spokesperson wording detected",
        "announced": "Announcement wording detected",
        "confirmed": "Confirmation wording detected",
        "documents show": "Document-based reporting wording detected",
        "interview": "Interview/reporting wording detected",
    }
    RISK_CUES = {
        "shocking": "Sensational wording detected",
        "secret": "Secret-claim wording detected",
        "miracle": "Miracle-claim wording detected",
        "conspiracy": "Conspiracy wording detected",
        "do not want you to know": "Clickbait/conspiracy wording detected",
        "won't believe": "Clickbait wording detected",
        "you will not believe": "Clickbait wording detected",
        "viral post": "Low-trust viral-post wording detected",
        "exposed": "Sensational wording detected",
        "hidden truth": "Hidden-truth wording detected",
        "hoax": "Hoax wording detected",
        "breaking!!!": "Excessive punctuation detected",
        "guaranteed": "Overclaim wording detected",
        "unbelievable": "Sensational wording detected",
    }

    def __init__(self, dataset_path: Path, artifact_dir: Path) -> None:
        self.dataset_path = dataset_path.resolve()
        self.artifact_dir = artifact_dir
        self.model_path = artifact_dir / "fake_news_pipeline.joblib"
        self.metrics_path = artifact_dir / "metrics.json"
        self.history_path = artifact_dir / "history.db"
        self.stemmer = PorterStemmer()
        self.stop_words = _load_stop_words()
        self.dataset_column_overrides = {
            "title": os.getenv("DATASET_TITLE_COLUMN", "").strip(),
            "author": os.getenv("DATASET_AUTHOR_COLUMN", "").strip(),
            "text": os.getenv("DATASET_TEXT_COLUMN", "").strip(),
            "label": os.getenv("DATASET_LABEL_COLUMN", "").strip(),
        }
        self.real_label_aliases = self._parse_label_aliases(
            os.getenv("DATASET_REAL_LABELS"),
            {"0", "0.0", "real", "true", "reliable", "legit", "genuine", "credible"},
        )
        self.fake_label_aliases = self._parse_label_aliases(
            os.getenv("DATASET_FAKE_LABELS"),
            {"1", "1.0", "fake", "false", "unreliable", "hoax", "rumor", "misleading"},
        )
        self._artifacts: TrainingArtifacts | None = None
        self._transformer_assets: tuple[Any, Any, Path] | None | bool = None
        self._ensure_history_table()

    def ensure_ready(self) -> dict[str, Any]:
        return self._load_or_train().metrics

    def safe_metrics(self) -> dict[str, Any]:
        try:
            return self.ensure_ready()
        except FileNotFoundError:
            return {
                "train_accuracy": "--",
                "test_accuracy": "--",
                "dataset_rows": 0,
                "features": 0,
                "meta_features": 27,
                "model": "Dataset missing",
                "secondary_model": "Unavailable",
            }

    def _parse_label_aliases(
        self,
        raw_value: str | None,
        defaults: set[str],
    ) -> set[str]:
        if not raw_value:
            return set(defaults)
        aliases = {
            self._normalize_label_token(item)
            for item in raw_value.split(",")
            if item.strip()
        }
        return aliases or set(defaults)

    def _normalize_label_token(self, value: object) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())

    def _training_signature(self) -> dict[str, object]:
        dataset_signature: dict[str, object] = {
            "path": str(self.dataset_path),
            "exists": self.dataset_path.exists(),
        }
        if self.dataset_path.exists():
            stat = self.dataset_path.stat()
            dataset_signature["size"] = stat.st_size
            dataset_signature["modified_ns"] = stat.st_mtime_ns

        return {
            "dataset": dataset_signature,
            "columns": self.dataset_column_overrides,
            "real_labels": sorted(self.real_label_aliases),
            "fake_labels": sorted(self.fake_label_aliases),
        }

    def _training_signatures_match(
        self,
        saved_signature: dict[str, object] | None,
        current_signature: dict[str, object],
    ) -> bool:
        if saved_signature == current_signature:
            return True
        if not isinstance(saved_signature, dict):
            return False
        if saved_signature.get("columns") != current_signature.get("columns"):
            return False
        if saved_signature.get("real_labels") != current_signature.get("real_labels"):
            return False
        if saved_signature.get("fake_labels") != current_signature.get("fake_labels"):
            return False

        saved_dataset = saved_signature.get("dataset")
        current_dataset = current_signature.get("dataset")
        if not isinstance(saved_dataset, dict) or not isinstance(current_dataset, dict):
            return False

        for field in ("exists", "size", "modified_ns"):
            if saved_dataset.get(field) != current_dataset.get(field):
                return False

        saved_name = Path(str(saved_dataset.get("path", ""))).name.casefold()
        current_name = Path(str(current_dataset.get("path", ""))).name.casefold()
        if saved_name and current_name and saved_name != current_name:
            return False
        return True

    def _read_dataset_frame(self) -> pd.DataFrame:
        suffix = self.dataset_path.suffix.lower()
        if suffix == ".json":
            frame = pd.read_json(self.dataset_path)
        elif suffix == ".jsonl":
            frame = pd.read_json(self.dataset_path, lines=True)
        elif suffix in {".tsv", ".txt"}:
            frame = pd.read_csv(self.dataset_path, sep="\t")
        else:
            frame = pd.read_csv(self.dataset_path)
        return frame.fillna("")

    def _prepare_training_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        title_column = self._resolve_dataset_column(
            dataset,
            "title",
            ["title", "headline", "news_title", "heading", "subject"],
            required=False,
        )
        author_column = self._resolve_dataset_column(
            dataset,
            "author",
            ["author", "source", "publisher", "news_source", "byline", "publication"],
            required=False,
        )
        text_column = self._resolve_dataset_column(
            dataset,
            "text",
            ["text", "content", "article", "article_text", "body", "story", "news", "news_text"],
            required=True,
        )
        label_column = self._resolve_dataset_column(
            dataset,
            "label",
            ["label", "class", "target", "output", "category", "is_fake", "fake"],
            required=True,
        )

        prepared = pd.DataFrame()
        prepared["text"] = dataset[text_column].map(self._clean_dataset_text)
        prepared["title"] = (
            dataset[title_column].map(self._clean_dataset_text)
            if title_column
            else prepared["text"].map(self._derive_title_from_text)
        )
        prepared["author"] = (
            dataset[author_column].map(self._clean_dataset_text)
            if author_column
            else ""
        )
        prepared["label"] = dataset[label_column].map(self._normalize_dataset_label)

        prepared = prepared[prepared["label"].isin([0, 1])].copy()
        prepared["label"] = prepared["label"].astype(int)
        prepared = prepared[prepared["text"].str.strip().ne("")].copy()
        prepared["title"] = prepared.apply(
            lambda row: row["title"] or self._derive_title_from_text(row["text"]),
            axis=1,
        )
        prepared["author"] = prepared["author"].fillna("").astype(str)
        prepared = prepared.drop_duplicates(
            subset=["title", "author", "text", "label"]
        ).reset_index(drop=True)

        if len(prepared) < 20:
            raise ValueError(
                "The new dataset did not produce enough usable rows after column/label normalization."
            )

        column_map = {
            "title": title_column or "(generated from text)",
            "author": author_column or "(not provided)",
            "text": text_column,
            "label": label_column,
        }
        return prepared, column_map

    def _resolve_dataset_column(
        self,
        dataset: pd.DataFrame,
        logical_name: str,
        aliases: list[str],
        required: bool,
    ) -> str | None:
        override = self.dataset_column_overrides.get(logical_name, "")
        if override:
            match = self._match_dataset_column(dataset.columns, override)
            if match is None:
                raise ValueError(
                    f"Dataset column override `{override}` for `{logical_name}` was not found."
                )
            return match

        for alias in aliases:
            match = self._match_dataset_column(dataset.columns, alias)
            if match is not None:
                return match

        if required:
            raise ValueError(
                f"Could not find a `{logical_name}` column in the dataset. "
                f"Available columns: {', '.join(str(column) for column in dataset.columns)}"
            )
        return None

    def _match_dataset_column(self, columns: Any, candidate: str) -> str | None:
        normalized_candidate = self._normalize_column_name(candidate)
        for column in columns:
            if self._normalize_column_name(str(column)) == normalized_candidate:
                return str(column)
        return None

    def _normalize_column_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _clean_dataset_text(self, value: object) -> str:
        return " ".join(str(value or "").split())

    def _derive_title_from_text(self, text: str) -> str:
        return " ".join(str(text).split()[:18]).strip()

    def _normalize_dataset_label(self, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not pd.isna(value):
            numeric = float(value)
            if numeric in {0.0, 1.0}:
                return int(numeric)

        token = self._normalize_label_token(value)
        if not token:
            return None
        if token in self.real_label_aliases:
            return 0
        if token in self.fake_label_aliases:
            return 1
        try:
            numeric = float(token)
        except ValueError:
            return None
        if numeric in {0.0, 1.0}:
            return int(numeric)
        return None
    def predict(
        self,
        title: str,
        author: str,
        text: str,
        url: str = "",
        source: str = "",
        published_at: str = "",
    ) -> dict[str, Any]:
        artifacts = self._load_or_train()
        source_profile = self._build_source_profile(url=url, source=source, author=author)
        weak_input = self._is_weak_input(title=title, author=author, text=text)
        raw_input = " ".join(part for part in [author, title, text] if part).strip()

        headline_content = self._build_headline_content(title=title, author=author, text=text)
        body_content = self._build_body_content(text)
        headline_probs = artifacts.headline_pipeline.predict_proba([headline_content])[0]
        body_probs = None
        if text.strip():
            body_probs = artifacts.body_pipeline.predict_proba([body_content])[0]

        signal_analysis = self._analyze_signals(
            title=title,
            author=author,
            text=text,
            source_profile=source_profile,
            published_at=published_at,
        )
        fusion_row = self._build_fusion_feature_row(
            title=title,
            author=author,
            text=text,
            headline_probs=headline_probs,
            body_probs=body_probs,
            weak_input=weak_input,
            signal_analysis=signal_analysis,
            source_profile=source_profile,
            published_at=published_at,
        )
        fusion_probs = artifacts.fusion_classifier.predict_proba([fusion_row])[0]
        transformer_result = self._transformer_predict(title=title, author=author, text=text)
        adjusted_probs = self._apply_context_adjustments(
            model_probs=(float(fusion_probs[0]), float(fusion_probs[1])),
            signal_analysis=signal_analysis,
            source_profile=source_profile,
            weak_input=weak_input,
            published_at=published_at,
            transformer_result=transformer_result,
        )
        decision = self._decide_label(
            adjusted_probs=adjusted_probs,
            headline_probs=headline_probs,
            body_probs=body_probs,
            transformer_result=transformer_result,
            signal_analysis=signal_analysis,
            weak_input=weak_input,
            source_profile=source_profile,
        )

        result = {
            "label": decision["label"],
            "label_code": decision["label_code"],
            "confidence": decision["confidence"],
            "probabilities": {
                "real": round(float(adjusted_probs[0]) * 100, 2),
                "fake": round(float(adjusted_probs[1]) * 100, 2),
            },
            "metrics": artifacts.metrics,
            "input_quality": {
                "is_weak": weak_input,
                "characters": len(raw_input),
                "words": len(raw_input.split()),
                "recommendation": decision["recommendation"],
            },
            "analysis": {
                "source_cues": signal_analysis["source_cues"],
                "reporting_cues": signal_analysis["reporting_cues"],
                "risk_cues": signal_analysis["risk_cues"],
                "metadata_cues": signal_analysis["metadata_cues"],
                "model_consensus": decision["consensus"],
                "secondary_model": (
                    transformer_result["model"] if transformer_result else "Unavailable"
                ),
            },
            "source_profile": {
                "label": source_profile["label"],
                "domain": source_profile["domain"],
                "score": round(source_profile["raw_score"] * 100, 2),
                "tier": source_profile["tier"],
                "kind": source_profile["kind"],
                "reason": source_profile["reason"],
            },
            "verification_links": self._build_verification_links(
                title=title,
                source_profile=source_profile,
                url=url,
            ),
            "model_breakdown": {
                "headline_model": self._format_probability_block(headline_probs),
                "body_model": (
                    self._format_probability_block(body_probs) if body_probs is not None else None
                ),
                "fusion_model": {
                    "real": round(float(fusion_probs[0]) * 100, 2),
                    "fake": round(float(fusion_probs[1]) * 100, 2),
                },
                "secondary_model": transformer_result,
            },
        }

        self._save_prediction(
            title=title,
            author=author,
            text=text,
            label=result["label"],
            confidence=result["confidence"],
            source=source_profile["label"],
            source_domain=source_profile["domain"],
            published_at=published_at,
            url=url,
            input_mode="url" if url else "manual",
            source_score=result["source_profile"]["score"],
            consensus=decision["consensus"],
            recommendation=decision["recommendation"],
        )
        return result

    def _saved_payload_is_usable(self, payload: Any) -> bool:
        return isinstance(payload, dict) and {"headline", "body", "fusion"} <= set(payload)

    def _read_saved_metrics(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            return {}
        try:
            payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_loaded_metrics(
        self,
        saved_metrics: dict[str, Any] | None,
        training_signature: dict[str, object],
        *,
        model_label: str,
        signature_match: bool,
    ) -> dict[str, Any]:
        metrics = dict(saved_metrics or {})
        metrics.setdefault("train_accuracy", None)
        metrics.setdefault("test_accuracy", None)
        metrics.setdefault("stack_validation_accuracy", None)
        metrics.setdefault("headline_test_accuracy", None)
        metrics.setdefault("body_test_accuracy", None)
        metrics.setdefault("headline_train_accuracy", None)
        metrics.setdefault("body_train_accuracy", None)
        metrics.setdefault("dataset_rows", 0)
        metrics.setdefault(
            "dataset_file",
            self.dataset_path.name if self.dataset_path.exists() else "",
        )
        metrics.setdefault("dataset_columns", {})
        metrics.setdefault("features", 0)
        metrics.setdefault("meta_features", 27)
        metrics["model"] = str(metrics.get("model") or model_label)
        metrics["model_version"] = str(metrics.get("model_version") or self.MODEL_VERSION)
        metrics["training_signature"] = training_signature
        metrics["training_signature_match"] = signature_match
        metrics["secondary_model"] = str(
            metrics.get("secondary_model")
            or (
                "BERT Tiny chunked second opinion available locally"
                if self._resolve_transformer_snapshot() is not None
                else "Unavailable"
            )
        )
        metrics["fusion_strategy"] = str(
            metrics.get("fusion_strategy")
            or "stacked headline/body/meta ensemble with source credibility signals"
        )
        return metrics

    def _build_training_artifacts(
        self,
        payload: dict[str, Any],
        metrics: dict[str, Any],
    ) -> TrainingArtifacts:
        return TrainingArtifacts(
            headline_pipeline=payload["headline"],
            body_pipeline=payload["body"],
            fusion_classifier=payload["fusion"],
            metrics=metrics,
        )

    def _build_heuristic_artifacts(self, reason: str) -> TrainingArtifacts:
        metrics = {
            "train_accuracy": None,
            "test_accuracy": None,
            "dataset_rows": 0,
            "dataset_file": self.dataset_path.name if self.dataset_path.exists() else "",
            "dataset_columns": {},
            "features": 0,
            "meta_features": 27,
            "model": "Heuristic fallback (deployment-safe mode)",
            "model_version": self.MODEL_VERSION,
            "training_signature": self._training_signature(),
            "secondary_model": (
                "BERT Tiny chunked second opinion available locally"
                if self._resolve_transformer_snapshot() is not None
                else "Unavailable"
            ),
            "fusion_strategy": reason,
            "runtime_mode": "heuristic_fallback",
        }
        return TrainingArtifacts(
            headline_pipeline=_HeuristicTextPipeline(self, mode="headline"),
            body_pipeline=_HeuristicTextPipeline(self, mode="body"),
            fusion_classifier=_HeuristicFusionClassifier(self),
            metrics=metrics,
        )

    def _load_or_train(self) -> TrainingArtifacts:
        if self._artifacts is not None:
            return self._artifacts

        current_training_signature = self._training_signature()
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
            except Exception:
                payload = None
            if self._saved_payload_is_usable(payload):
                saved_metrics = self._read_saved_metrics()
                signature_match = self._training_signatures_match(
                    saved_metrics.get("training_signature"),
                    current_training_signature,
                )
                metrics = self._build_loaded_metrics(
                    saved_metrics=saved_metrics,
                    training_signature=current_training_signature,
                    model_label=(
                        saved_metrics.get("model")
                        if signature_match and saved_metrics.get("model")
                        else "Loaded pre-trained model artifact"
                    ),
                    signature_match=signature_match,
                )
                try:
                    self.artifact_dir.mkdir(parents=True, exist_ok=True)
                    self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                except OSError:
                    pass
                self._artifacts = self._build_training_artifacts(payload, metrics)
                return self._artifacts

        if self.dataset_path.exists():
            self._artifacts = self._train()
            return self._artifacts

        self._artifacts = self._build_heuristic_artifacts(
            "No dataset or trained model artifact was available, so the app is using"
            " a lightweight fallback scorer for deployment.",
        )
        return self._artifacts

    def _train(self) -> TrainingArtifacts:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Set DATASET_PATH or place your dataset file in the expected location."
            )

        dataset, dataset_columns = self._prepare_training_dataset(self._read_dataset_frame())
        dataset["headline_content"] = dataset.apply(
            lambda row: self._build_headline_content(
                title=str(row.get("title", "")),
                author=str(row.get("author", "")),
                text=str(row.get("text", "")),
            ),
            axis=1,
        )
        dataset["body_content"] = dataset["text"].apply(self._build_body_content)

        indices = dataset.index.values
        labels = dataset["label"].values
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=2,
        )
        train_labels = dataset.loc[train_idx, "label"].values
        base_train_idx, stack_idx = train_test_split(
            train_idx,
            test_size=0.25,
            stratify=train_labels,
            random_state=17,
        )
        y_stack = dataset.loc[stack_idx, "label"].values
        y_train = dataset.loc[train_idx, "label"].values
        y_test = dataset.loc[test_idx, "label"].values

        stack_headline_pipeline = self._make_pipeline(max_features=45000)
        stack_body_pipeline = self._make_pipeline(max_features=35000)
        stack_headline_pipeline.fit(
            dataset.loc[base_train_idx, "headline_content"].values,
            dataset.loc[base_train_idx, "label"].values,
        )
        stack_body_pipeline.fit(
            dataset.loc[base_train_idx, "body_content"].values,
            dataset.loc[base_train_idx, "label"].values,
        )
        stack_headline_probs = stack_headline_pipeline.predict_proba(
            dataset.loc[stack_idx, "headline_content"].values
        )
        stack_body_probs = stack_body_pipeline.predict_proba(
            dataset.loc[stack_idx, "body_content"].values
        )
        stack_rows = self._build_fusion_matrix(
            frame=dataset.loc[stack_idx],
            headline_probs=stack_headline_probs,
            body_probs=stack_body_probs,
        )

        fusion_classifier = self._make_fusion_classifier()
        fusion_classifier.fit(stack_rows, y_stack)
        stack_validation_probs = fusion_classifier.predict_proba(stack_rows)
        stack_validation_accuracy = accuracy_score(
            y_stack,
            (stack_validation_probs[:, 1] >= 0.5).astype(int),
        )
        headline_pipeline = self._make_pipeline(max_features=45000)
        body_pipeline = self._make_pipeline(max_features=35000)
        headline_pipeline.fit(dataset.loc[train_idx, "headline_content"].values, y_train)
        body_pipeline.fit(dataset.loc[train_idx, "body_content"].values, y_train)

        headline_train_probs = headline_pipeline.predict_proba(
            dataset.loc[train_idx, "headline_content"].values
        )
        body_train_probs = body_pipeline.predict_proba(
            dataset.loc[train_idx, "body_content"].values
        )
        headline_test_probs = headline_pipeline.predict_proba(
            dataset.loc[test_idx, "headline_content"].values
        )
        body_test_probs = body_pipeline.predict_proba(
            dataset.loc[test_idx, "body_content"].values
        )

        train_rows = self._build_fusion_matrix(
            frame=dataset.loc[train_idx],
            headline_probs=headline_train_probs,
            body_probs=body_train_probs,
        )
        test_rows = self._build_fusion_matrix(
            frame=dataset.loc[test_idx],
            headline_probs=headline_test_probs,
            body_probs=body_test_probs,
        )
        fusion_train_probs = fusion_classifier.predict_proba(train_rows)
        fusion_test_probs = fusion_classifier.predict_proba(test_rows)

        headline_train_accuracy = accuracy_score(
            y_train,
            headline_pipeline.predict(dataset.loc[train_idx, "headline_content"].values),
        )
        body_train_accuracy = accuracy_score(
            y_train,
            body_pipeline.predict(dataset.loc[train_idx, "body_content"].values),
        )
        fusion_train_accuracy = accuracy_score(
            y_train,
            (fusion_train_probs[:, 1] >= 0.5).astype(int),
        )
        headline_test_accuracy = accuracy_score(
            y_test,
            headline_pipeline.predict(dataset.loc[test_idx, "headline_content"].values),
        )
        body_test_accuracy = accuracy_score(
            y_test,
            body_pipeline.predict(dataset.loc[test_idx, "body_content"].values),
        )
        fusion_test_accuracy = accuracy_score(
            y_test,
            (fusion_test_probs[:, 1] >= 0.5).astype(int),
        )

        transformer_path = self._resolve_transformer_snapshot()
        metrics = {
            "train_accuracy": round(float(fusion_train_accuracy) * 100, 2),
            "test_accuracy": round(float(fusion_test_accuracy) * 100, 2),
            "stack_validation_accuracy": round(float(stack_validation_accuracy) * 100, 2),
            "headline_test_accuracy": round(float(headline_test_accuracy) * 100, 2),
            "body_test_accuracy": round(float(body_test_accuracy) * 100, 2),
            "headline_train_accuracy": round(float(headline_train_accuracy) * 100, 2),
            "body_train_accuracy": round(float(body_train_accuracy) * 100, 2),
            "dataset_rows": int(len(dataset)),
            "dataset_file": self.dataset_path.name,
            "dataset_columns": dataset_columns,
            "features": int(
                len(headline_pipeline.named_steps["tfidf"].get_feature_names_out())
                + len(body_pipeline.named_steps["tfidf"].get_feature_names_out())
            ),
            "meta_features": 27,
            "model": "Stacked Hybrid Linear SVC + TF-IDF",
            "model_version": self.MODEL_VERSION,
            "training_signature": self._training_signature(),
            "secondary_model": (
                "BERT Tiny chunked second opinion available locally"
                if transformer_path is not None
                else "Unavailable"
            ),
            "fusion_strategy": "stacked headline/body/meta ensemble with source credibility signals",
        }

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"headline": headline_pipeline, "body": body_pipeline, "fusion": fusion_classifier},
            self.model_path,
        )
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return TrainingArtifacts(
            headline_pipeline=headline_pipeline,
            body_pipeline=body_pipeline,
            fusion_classifier=fusion_classifier,
            metrics=metrics,
        )

    def _make_pipeline(self, max_features: int) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced"),
                        cv=3,
                    ),
                ),
            ]
        )

    def _make_fusion_classifier(self) -> LogisticRegression:
        return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")

    def get_prediction_history(self, limit: int = 10) -> list[dict[str, Any]]:
        with sqlite3.connect(self.history_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT created_at, title, author, label, confidence, source, source_domain,
                       published_at, url, input_mode, source_score, consensus, recommendation
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _build_headline_content(self, title: str, author: str, text: str) -> str:
        shortened_text = " ".join(text.split()[:160])
        title_block = " ".join(filter(None, [title, title]))
        author_block = " ".join(filter(None, [author, author]))
        return self._stem_text(" ".join(part for part in [author_block, title_block, shortened_text] if part))

    def _build_body_content(self, text: str) -> str:
        return self._stem_text(" ".join(text.split()[:420]))

    def _stem_text(self, content: str) -> str:
        letters_only = re.sub(r"[^a-zA-Z]", " ", content)
        tokens = letters_only.lower().split()
        normalized = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return " ".join(normalized)

    def _is_weak_input(self, title: str, author: str, text: str) -> bool:
        title_words = len(title.split())
        author_words = len(author.split())
        text_words = len(text.split())
        combined_words = title_words + author_words + text_words
        return text_words < 45 and (combined_words < 18 or title_words < 6)

    def _build_fusion_matrix(self, frame: pd.DataFrame, headline_probs: Any, body_probs: Any) -> list[list[float]]:
        rows: list[list[float]] = []
        for row, headline_row, body_row in zip(frame.itertuples(index=False), headline_probs, body_probs):
            title = str(getattr(row, "title", ""))
            author = str(getattr(row, "author", ""))
            text = str(getattr(row, "text", ""))
            source_profile = self._build_source_profile(url="", source=author, author=author)
            weak_input = self._is_weak_input(title=title, author=author, text=text)
            signal_analysis = self._analyze_signals(
                title=title,
                author=author,
                text=text,
                source_profile=source_profile,
                published_at="",
            )
            rows.append(
                self._build_fusion_feature_row(
                    title=title,
                    author=author,
                    text=text,
                    headline_probs=headline_row,
                    body_probs=body_row,
                    weak_input=weak_input,
                    signal_analysis=signal_analysis,
                    source_profile=source_profile,
                    published_at="",
                )
            )
        return rows

    def _build_fusion_feature_row(
        self,
        title: str,
        author: str,
        text: str,
        headline_probs: Any,
        body_probs: Any | None,
        weak_input: bool,
        signal_analysis: dict[str, Any],
        source_profile: dict[str, Any],
        published_at: str,
    ) -> list[float]:
        title_words = len(title.split())
        author_words = len(author.split())
        text_words = len(text.split())
        combined_words = title_words + author_words + text_words
        fallback_body = body_probs if body_probs is not None else headline_probs
        combined = " ".join(part for part in [title, author, text] if part).strip()
        return [
            float(headline_probs[0]),
            float(headline_probs[1]),
            float(fallback_body[0]),
            float(fallback_body[1]),
            1.0 if body_probs is not None and text_words > 0 else 0.0,
            abs(float(headline_probs[1]) - float(fallback_body[1])),
            min(1.0, log1p(title_words) / log1p(30)),
            1.0 if author_words > 0 else 0.0,
            min(1.0, author_words / 5.0),
            min(1.0, log1p(text_words) / log1p(900)),
            min(1.0, log1p(combined_words) / log1p(950)),
            min(1.0, combined.count("!") / 4.0),
            min(1.0, combined.count("?") / 3.0),
            self._uppercase_ratio(combined),
            self._digit_ratio(combined),
            self._lexical_diversity(combined),
            min(1.0, self._average_word_length(combined) / 8.0),
            self._title_body_overlap(title=title, text=text),
            source_profile["raw_score"],
            1.0 if source_profile["known"] else 0.0,
            1.0 if source_profile["kind"] == "platform" else 0.0,
            1.0 if published_at else 0.0,
            min(1.0, len(signal_analysis["source_cues"]) / 3.0),
            min(1.0, len(signal_analysis["reporting_cues"]) / 5.0),
            min(1.0, len(signal_analysis["risk_cues"]) / 5.0),
            1.0 if weak_input else 0.0,
            1.0 if text_words >= 180 else 0.0,
        ]

    def _heuristic_text_probabilities(
        self,
        sample: str,
        mode: str,
    ) -> tuple[float, float]:
        content = str(sample or "")
        lowered = content.lower()
        words = lowered.split()
        word_count = len(words)
        source_hits = len(self._find_cues(lowered, self.SOURCE_CUES))
        reporting_hits = len(self._find_cues(lowered, self.REPORTING_CUES))
        risk_hits = len(self._find_cues(lowered, self.RISK_CUES))

        real_score = 0.5
        fake_score = 0.5
        real_score += min(0.18, source_hits * 0.06)
        real_score += min(0.14, reporting_hits * 0.04)
        fake_score += min(0.24, risk_hits * 0.07)
        fake_score += min(0.08, content.count("!") * 0.02)
        fake_score += min(0.05, content.count("?") * 0.02)
        fake_score += min(0.10, self._uppercase_ratio(content) * 0.25)
        fake_score += min(0.08, self._digit_ratio(content) * 0.40)

        if word_count >= 80:
            real_score += 0.04
        if word_count < 25:
            fake_score += 0.04
        if mode == "body" and word_count >= 120:
            real_score += 0.08
        if mode == "headline" and word_count < 12:
            fake_score += 0.05

        return self._normalize_pair(max(real_score, 0.01), max(fake_score, 0.01))

    def _heuristic_fusion_probabilities(self, row: list[float]) -> tuple[float, float]:
        headline_real, headline_fake = float(row[0]), float(row[1])
        body_real, body_fake = float(row[2]), float(row[3])
        disagreement = float(row[5])
        exclamation_ratio = float(row[11])
        question_ratio = float(row[12])
        uppercase_ratio = float(row[13])
        source_score = float(row[18])
        platform_flag = float(row[20])
        source_cues = float(row[22])
        reporting_cues = float(row[23])
        risk_cues = float(row[24])
        weak_input = float(row[25])
        long_body = float(row[26])

        real_score = (
            0.32
            + (headline_real * 0.22)
            + (body_real * 0.28)
            + (source_score * 0.12)
            + (source_cues * 0.05)
            + (reporting_cues * 0.08)
            + (long_body * 0.04)
        )
        fake_score = (
            0.32
            + (headline_fake * 0.22)
            + (body_fake * 0.28)
            + (risk_cues * 0.10)
            + (platform_flag * 0.08)
            + (exclamation_ratio * 0.05)
            + (question_ratio * 0.03)
            + (uppercase_ratio * 0.05)
            + (weak_input * 0.04)
            + (disagreement * 0.05)
        )

        real_score += max(0.0, (source_score - 0.5) * 0.12)
        fake_score += max(0.0, (0.55 - source_score) * 0.18)

        return self._normalize_pair(max(real_score, 0.01), max(fake_score, 0.01))

    def _apply_context_adjustments(
        self,
        model_probs: tuple[float, float],
        signal_analysis: dict[str, Any],
        source_profile: dict[str, Any],
        weak_input: bool,
        published_at: str,
        transformer_result: dict[str, Any] | None,
    ) -> tuple[float, float]:
        real_prob, fake_prob = model_probs
        if transformer_result is not None:
            weight = 0.18 if transformer_result.get("chunk_count", 1) > 1 else 0.12
            transformer_real = transformer_result["probabilities"]["real"] / 100
            transformer_fake = transformer_result["probabilities"]["fake"] / 100
            real_prob = (real_prob * (1 - weight)) + (transformer_real * weight)
            fake_prob = (fake_prob * (1 - weight)) + (transformer_fake * weight)

        reputation_delta = source_profile["raw_score"] - 0.5
        if reputation_delta > 0:
            real_prob += min(0.08, reputation_delta * 0.18)
        if source_profile["kind"] == "platform":
            fake_prob += 0.05
        if source_profile["kind"] == "partisan":
            fake_prob += 0.03
        if published_at:
            real_prob += 0.03

        real_prob += min(0.05, len(signal_analysis["source_cues"]) * 0.015)
        real_prob += min(0.05, len(signal_analysis["reporting_cues"]) * 0.012)
        fake_prob += min(0.10, len(signal_analysis["risk_cues"]) * 0.03)

        if weak_input:
            real_prob *= 0.985
            fake_prob *= 0.985
        return self._normalize_pair(real_prob, fake_prob)

    def _decide_label(
        self,
        adjusted_probs: tuple[float, float],
        headline_probs: Any,
        body_probs: Any | None,
        transformer_result: dict[str, Any] | None,
        signal_analysis: dict[str, Any],
        weak_input: bool,
        source_profile: dict[str, Any],
    ) -> dict[str, Any]:
        real_prob, fake_prob = adjusted_probs
        source_cues = signal_analysis["source_cues"]
        reporting_cues = signal_analysis["reporting_cues"]
        risk_cues = signal_analysis["risk_cues"]
        disagreement = abs(float(headline_probs[1]) - float(body_probs[1])) if body_probs is not None else 0.0
        transformer_conflict = False
        if transformer_result is not None:
            transformer_conflict = (
                int(transformer_result["label_code"]) != int(fake_prob >= real_prob)
                and transformer_result["confidence"] >= 88
            )

        if weak_input and max(real_prob, fake_prob) < 0.84:
            return {"label": "Needs More Context", "label_code": -1, "confidence": round(max(real_prob, fake_prob) * 100, 2), "recommendation": "This input is too short for a trustworthy verdict. Add more article text, a source URL, or a publish date.", "consensus": "weak input"}
        if disagreement > 0.35 and max(real_prob, fake_prob) < 0.9:
            return {"label": "Needs More Context", "label_code": -1, "confidence": round(max(real_prob, fake_prob) * 100, 2), "recommendation": "Headline cues and article-body cues disagree. Paste more of the article or verify against the original source.", "consensus": "headline/body disagreement"}
        if transformer_conflict and max(real_prob, fake_prob) < 0.92:
            return {"label": "Needs More Context", "label_code": -1, "confidence": round(max(real_prob, fake_prob) * 100, 2), "recommendation": "The chunked language model disagrees with the primary ensemble. Verify this claim before trusting the verdict.", "consensus": "secondary-model disagreement"}
        if source_profile["kind"] == "platform" and not reporting_cues and fake_prob < 0.8:
            return {"label": "Needs More Context", "label_code": -1, "confidence": round(max(real_prob, fake_prob) * 100, 2), "recommendation": "This looks like user-generated or platform-first content. Check the original reporting source before deciding.", "consensus": "platform source"}
        if source_profile["raw_score"] >= 0.82 and real_prob >= 0.58 and not risk_cues:
            return {"label": "Real", "label_code": 0, "confidence": round(real_prob * 100, 2), "recommendation": "Trusted-source signals and reporting-style language support a real-news verdict.", "consensus": "source-supported real"}
        if risk_cues and fake_prob >= 0.68 and source_profile["raw_score"] < 0.75:
            return {"label": "Fake", "label_code": 1, "confidence": round(fake_prob * 100, 2), "recommendation": "Sensational or clickbait language pushed the verdict toward fake news.", "consensus": "risk cues aligned"}
        if real_prob >= 0.74 and (reporting_cues or source_profile["raw_score"] >= 0.6 or body_probs is not None):
            return {"label": "Real", "label_code": 0, "confidence": round(real_prob * 100, 2), "recommendation": "The article is consistent with legitimate reporting and source signals.", "consensus": "ensemble leans real"}
        if fake_prob >= 0.82 and source_profile["raw_score"] < 0.65:
            return {"label": "Fake", "label_code": 1, "confidence": round(fake_prob * 100, 2), "recommendation": "The ensemble strongly matches fake-news patterns and weak source credibility.", "consensus": "ensemble leans fake"}
        if abs(real_prob - fake_prob) < 0.12:
            return {"label": "Needs More Context", "label_code": -1, "confidence": round(max(real_prob, fake_prob) * 100, 2), "recommendation": "The result is too close to call. Check fact-checking sites or compare with trusted coverage.", "consensus": "close call"}
        if not source_cues and not reporting_cues and max(real_prob, fake_prob) < 0.76:
            return {"label": "Needs More Context", "label_code": -1, "confidence": round(max(real_prob, fake_prob) * 100, 2), "recommendation": "No trusted source or reporting cues were found. Add more article context or verify with external sources.", "consensus": "no source evidence"}
        if fake_prob >= real_prob:
            return {"label": "Fake", "label_code": 1, "confidence": round(fake_prob * 100, 2), "recommendation": "The model leans fake, but you should still verify against trusted reporting.", "consensus": "probability lead"}
        return {"label": "Real", "label_code": 0, "confidence": round(real_prob * 100, 2), "recommendation": "The model leans real, but you should still verify with the original reporting source.", "consensus": "probability lead"}

    def _analyze_signals(self, title: str, author: str, text: str, source_profile: dict[str, Any], published_at: str) -> dict[str, Any]:
        combined = " ".join(part for part in [title, author, text, source_profile["label"], source_profile["domain"]] if part).lower()
        source_cues = self._find_cues(combined, self.SOURCE_CUES)
        reporting_cues = self._find_cues(combined, self.REPORTING_CUES)
        risk_cues = self._find_cues(combined, self.RISK_CUES)
        if combined.count("!") >= 2:
            risk_cues.append("Multiple exclamation marks detected")
        if self._uppercase_ratio(" ".join([title, text])) > 0.22:
            risk_cues.append("Unusually high uppercase ratio detected")

        metadata_cues: list[str] = []
        if source_profile["known"] and source_profile["raw_score"] >= 0.75:
            metadata_cues.append("Known newsroom/source matched")
        elif source_profile["known"]:
            metadata_cues.append("Known source family matched")
        if published_at:
            metadata_cues.append("Publish date available")
        if len(text.split()) >= 180:
            metadata_cues.append("Long article body provided")
        if self._title_body_overlap(title=title, text=text) >= 0.18:
            metadata_cues.append("Headline terms align with article body")
        return {
            "source_cues": self._dedupe(source_cues),
            "reporting_cues": self._dedupe(reporting_cues),
            "risk_cues": self._dedupe(risk_cues),
            "metadata_cues": self._dedupe(metadata_cues),
        }

    def _find_cues(self, content: str, mapping: dict[str, str]) -> list[str]:
        hits: list[str] = []
        for needle, description in mapping.items():
            pattern = r"\b" + re.escape(needle).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, content):
                hits.append(description)
        return hits

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                unique.append(value)
        return unique

    def _build_source_profile(self, url: str, source: str, author: str) -> dict[str, Any]:
        domain = self._extract_domain(url)
        search_text = " ".join(part for part in [source, author] if part).lower()
        for token, label, score, kind in self.SOURCE_PROFILE_RULES:
            if "." in token and domain.endswith(token):
                return self._format_source_profile(label, domain, score, kind, True, f"Matched source domain `{token}`.")
        for token, label, score, kind in self.SOURCE_PROFILE_RULES:
            if "." not in token and token in search_text:
                return self._format_source_profile(label, domain, score, kind, True, f"Matched source name `{label}`.")
        label = source.strip() or author.strip() or domain or "Unknown source"
        reason = "No known source profile matched." if label != "Unknown source" else "No source information was supplied."
        return self._format_source_profile(label, domain, 0.46, "unknown", False, reason)

    def _format_source_profile(self, label: str, domain: str, score: float, kind: str, known: bool, reason: str) -> dict[str, Any]:
        if score >= 0.85:
            tier = "High credibility"
        elif score >= 0.65:
            tier = "Established outlet"
        elif score >= 0.45:
            tier = "Unverified source"
        else:
            tier = "Low-trust platform"
        return {"label": label, "domain": domain, "raw_score": score, "kind": kind, "known": known, "tier": tier, "reason": reason}

    def _extract_domain(self, url: str) -> str:
        hostname = urlparse(url).hostname or ""
        return hostname.lower().removeprefix("www.")

    def _title_body_overlap(self, title: str, text: str) -> float:
        title_tokens = {token for token in re.findall(r"[a-zA-Z]{3,}", title.lower()) if token not in self.stop_words}
        if not title_tokens:
            return 0.0
        text_tokens = {token for token in re.findall(r"[a-zA-Z]{3,}", text.lower()) if token not in self.stop_words}
        if not text_tokens:
            return 0.0
        return len(title_tokens & text_tokens) / max(1, len(title_tokens))

    def _uppercase_ratio(self, content: str) -> float:
        letters = [character for character in content if character.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for character in letters if character.isupper()) / len(letters)

    def _digit_ratio(self, content: str) -> float:
        return 0.0 if not content else sum(1 for character in content if character.isdigit()) / len(content)

    def _lexical_diversity(self, content: str) -> float:
        tokens = re.findall(r"[a-zA-Z]{3,}", content.lower())
        return 0.0 if not tokens else len(set(tokens)) / len(tokens)

    def _average_word_length(self, content: str) -> float:
        tokens = re.findall(r"[a-zA-Z]+", content)
        return 0.0 if not tokens else sum(len(token) for token in tokens) / len(tokens)
    def _resolve_transformer_snapshot(self) -> Path | None:
        ref_path = self.TRANSFORMER_REPO_DIR / "refs" / "main"
        if not ref_path.exists():
            return None
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = self.TRANSFORMER_REPO_DIR / "snapshots" / revision
        return snapshot_path if snapshot_path.exists() else None

    def _load_transformer_assets(self) -> tuple[Any, Any, Path] | None:
        if self._transformer_assets is False:
            return None
        if isinstance(self._transformer_assets, tuple):
            return self._transformer_assets
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self._transformer_assets = False
            return None

        snapshot_path = self._resolve_transformer_snapshot()
        if snapshot_path is None:
            self._transformer_assets = False
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
            model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)
            model.eval()
        except Exception:
            self._transformer_assets = False
            return None

        self._transformer_assets = (tokenizer, model, snapshot_path)
        return self._transformer_assets

    def _transformer_predict(self, title: str, author: str, text: str) -> dict[str, Any] | None:
        if len(text.split()) < 35:
            return None
        assets = self._load_transformer_assets()
        if assets is None or torch is None:
            return None

        tokenizer, model, _ = assets
        chunk_probabilities: list[list[float]] = []
        for chunk in self._build_text_chunks(text):
            sample = " ".join(part for part in [title, author, chunk] if part).strip()
            inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[0].tolist()
            chunk_probabilities.append(probabilities)

        if not chunk_probabilities:
            return None
        average_real = sum(row[0] for row in chunk_probabilities) / len(chunk_probabilities)
        average_fake = sum(row[1] for row in chunk_probabilities) / len(chunk_probabilities)
        average_real, average_fake = self._normalize_pair(average_real, average_fake)
        return {
            "model": "BERT Tiny chunked second opinion",
            "label_code": int(average_fake >= average_real),
            "confidence": round(max(average_real, average_fake) * 100, 2),
            "chunk_count": len(chunk_probabilities),
            "probabilities": {
                "real": round(average_real * 100, 2),
                "fake": round(average_fake * 100, 2),
            },
        }

    def _build_text_chunks(self, text: str, chunk_size: int = 180, overlap: int = 30, max_chunks: int = 4) -> list[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [" ".join(words)]
        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + chunk_size]
            if len(chunk) < 35:
                continue
            chunks.append(" ".join(chunk))
            if len(chunks) >= max_chunks:
                break
        return chunks

    def _build_verification_links(self, title: str, source_profile: dict[str, Any], url: str) -> list[dict[str, str]]:
        query_seed = " ".join(part for part in [title, source_profile["label"], source_profile["domain"]] if part).strip() or "news fact check"
        encoded = quote_plus(query_seed)
        links = [
            {"label": "Google fact-check search", "url": f"https://www.google.com/search?q={encoded}+fact+check"},
            {"label": "Search Snopes", "url": f"https://www.google.com/search?q=site%3Asnopes.com+{encoded}"},
            {"label": "Search Reuters Fact Check", "url": f"https://www.google.com/search?q=site%3Areuters.com+fact+check+{encoded}"},
            {"label": "Search PolitiFact", "url": f"https://www.google.com/search?q=site%3Apolitifact.com+{encoded}"},
        ]
        if url:
            links.insert(0, {"label": "Open original article", "url": url})
        return links

    def _format_probability_block(self, probabilities: Any) -> dict[str, float]:
        return {"real": round(float(probabilities[0]) * 100, 2), "fake": round(float(probabilities[1]) * 100, 2)}

    def _normalize_pair(self, real_prob: float, fake_prob: float) -> tuple[float, float]:
        total = real_prob + fake_prob
        return (0.5, 0.5) if total <= 0 else (real_prob / total, fake_prob / total)

    def _ensure_history_table(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.history_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
                """
            )
            existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_history)").fetchall()}
            migrations = {
                "source": "TEXT NOT NULL DEFAULT ''",
                "source_domain": "TEXT NOT NULL DEFAULT ''",
                "published_at": "TEXT NOT NULL DEFAULT ''",
                "url": "TEXT NOT NULL DEFAULT ''",
                "input_mode": "TEXT NOT NULL DEFAULT 'manual'",
                "source_score": "REAL NOT NULL DEFAULT 0",
                "consensus": "TEXT NOT NULL DEFAULT ''",
                "recommendation": "TEXT NOT NULL DEFAULT ''",
            }
            for column_name, column_sql in migrations.items():
                if column_name not in existing_columns:
                    connection.execute(f"ALTER TABLE prediction_history ADD COLUMN {column_name} {column_sql}")
            connection.commit()

    def _save_prediction(
        self,
        title: str,
        author: str,
        text: str,
        label: str,
        confidence: float,
        source: str,
        source_domain: str,
        published_at: str,
        url: str,
        input_mode: str,
        source_score: float,
        consensus: str,
        recommendation: str,
    ) -> None:
        with sqlite3.connect(self.history_path) as connection:
            connection.execute(
                """
                INSERT INTO prediction_history (
                    created_at, title, author, text, label, confidence, source, source_domain,
                    published_at, url, input_mode, source_score, consensus, recommendation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    title,
                    author,
                    text,
                    label,
                    confidence,
                    source,
                    source_domain,
                    published_at,
                    url,
                    input_mode,
                    source_score,
                    consensus,
                    recommendation,
                ),
            )
            connection.commit()
