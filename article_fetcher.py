from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

KNOWN_SOURCES = {
    "nytimes.com": "The New York Times",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "npr.org": "NPR",
    "cnn.com": "CNN",
    "theguardian.com": "The Guardian",
    "washingtonpost.com": "The Washington Post",
    "bloomberg.com": "Bloomberg",
    "wsj.com": "Wall Street Journal",
    "ft.com": "Financial Times",
}


@dataclass
class ExtractedArticle:
    url: str
    title: str
    author: str
    source: str
    domain: str
    published_at: str
    text: str


def fetch_article(url: str) -> ExtractedArticle:
    normalized_url = _normalize_url(url)
    normalized_path = urlparse(normalized_url).path.lower()
    if any(token in normalized_path for token in ("/audio/", "/video/", "/iplayer/", "/sounds/")):
        raise ValueError(
            "This looks like an audio or video page, not a text news article. Open the article or transcript page, or use Paste Article."
        )
    response = requests.get(normalized_url, headers=DEFAULT_HEADERS, timeout=12)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and "xml" not in content_type:
        raise ValueError("This URL does not look like a public webpage.")

    soup = BeautifulSoup(response.text, "html.parser")
    schema_items = _extract_schema_items(soup)
    title = _extract_title(soup, schema_items)
    author = _extract_author(soup, schema_items)
    source = _extract_source(soup, normalized_url, schema_items)
    domain = _extract_domain(normalized_url)
    published_at = _extract_published_at(soup, schema_items)
    text = _extract_text(soup, schema_items)

    if not text or len(text.split()) < 40:
        raise ValueError("Could not extract enough article text from this page. Try a direct news article URL.")

    return ExtractedArticle(
        url=normalized_url,
        title=_append_source(title, source),
        author=author or source or domain,
        source=source or domain,
        domain=domain,
        published_at=published_at,
        text=text,
    )


def _normalize_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("Please provide a news article URL.")
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or "." not in parsed.netloc:
        raise ValueError("Please enter a full article URL from a public website.")
    return cleaned


def _extract_title(soup: BeautifulSoup, schema_items: list[dict[str, Any]]) -> str:
    candidates = [
        ('meta[property="og:title"]', "content"),
        ('meta[name="twitter:title"]', "content"),
        ('meta[name="title"]', "content"),
    ]
    for selector, attr in candidates:
        node = soup.select_one(selector)
        value = _clean_text(node.get(attr, "")) if node else ""
        if value:
            return value
    for item in schema_items:
        for key in ("headline", "name"):
            value = _clean_text(_schema_text_value(item.get(key)))
            if value:
                return value
    return _clean_text(soup.title.string if soup.title and soup.title.string else "")


def _extract_author(soup: BeautifulSoup, schema_items: list[dict[str, Any]]) -> str:
    candidates = [
        ('meta[name="author"]', "content"),
        ('meta[property="article:author"]', "content"),
        ('meta[name="parsely-author"]', "content"),
        ('meta[name="dc.creator"]', "content"),
    ]
    for selector, attr in candidates:
        node = soup.select_one(selector)
        value = _clean_text(node.get(attr, "")) if node else ""
        if value:
            return value
    for item in schema_items:
        value = _clean_text(_schema_name_value(item.get("author")))
        if value:
            return value
    return ""


def _extract_source(soup: BeautifulSoup, url: str, schema_items: list[dict[str, Any]]) -> str:
    candidates = [
        ('meta[property="og:site_name"]', "content"),
        ('meta[name="application-name"]', "content"),
    ]
    for selector, attr in candidates:
        node = soup.select_one(selector)
        value = _clean_text(node.get(attr, "")) if node else ""
        if value:
            return value
    for item in schema_items:
        value = _clean_text(_schema_name_value(item.get("publisher")))
        if value:
            return value

    hostname = _extract_domain(url)
    for domain, label in KNOWN_SOURCES.items():
        if hostname.endswith(domain):
            return label
    return hostname


def _extract_published_at(soup: BeautifulSoup, schema_items: list[dict[str, Any]]) -> str:
    candidates = [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="article:published_time"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[name="publish-date"]', "content"),
        ('meta[name="parsely-pub-date"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ('time[datetime]', "datetime"),
    ]
    for selector, attr in candidates:
        node = soup.select_one(selector)
        value = _normalize_date_value(node.get(attr, "")) if node else ""
        if value:
            return value
    for item in schema_items:
        for key in ("datePublished", "dateCreated", "dateModified"):
            value = _normalize_date_value(_schema_text_value(item.get(key)))
            if value:
                return value
    return ""


def _extract_text(soup: BeautifulSoup, schema_items: list[dict[str, Any]]) -> str:
    for tag in soup(["script", "style", "noscript", "svg", "form", "header", "footer"]):
        tag.decompose()

    paragraphs: list[str] = []
    for item in schema_items:
        body = _schema_text_value(item.get("articleBody"))
        if body:
            paragraphs = _merge_paragraphs(paragraphs, _split_into_paragraphs(body))
    if len(" ".join(paragraphs).split()) >= 120:
        return " ".join(paragraphs)

    selectors = [
        "[itemprop='articleBody'] p",
        "[data-testid='article-body'] p",
        "[data-component='text-block'] p",
        "article p",
        "main p",
        "[role='main'] p",
        ".article-body p",
        ".article-content p",
        ".story-content p",
        ".entry-content p",
        ".post-content p",
        ".StoryBodyCompanionColumn p",
        ".caas-body p",
    ]
    for selector in selectors:
        if len(" ".join(paragraphs).split()) >= 80:
            break
        paragraphs = _merge_paragraphs(paragraphs, _collect_paragraphs(soup.select(selector)))

    if len(" ".join(paragraphs).split()) < 80:
        paragraphs = _merge_paragraphs(paragraphs, _collect_paragraphs(soup.find_all("p")))

    return " ".join(" ".join(paragraphs).split())


def _collect_paragraphs(nodes: Iterable[object]) -> list[str]:
    paragraphs: list[str] = []
    for node in nodes:
        text = _clean_text(node.get_text(" ", strip=True))
        if len(text.split()) >= 8:
            paragraphs.append(text)
    return paragraphs


def _merge_paragraphs(primary: list[str], secondary: list[str]) -> list[str]:
    seen = set(primary)
    merged = list(primary)
    for paragraph in secondary:
        if paragraph not in seen:
            merged.append(paragraph)
            seen.add(paragraph)
    return merged


def _append_source(title: str, source: str) -> str:
    cleaned_title = title.strip()
    cleaned_source = source.strip()
    if not cleaned_source or cleaned_source.lower() in cleaned_title.lower():
        return cleaned_title
    return f"{cleaned_title} - {cleaned_source}"


def _extract_schema_items(soup: BeautifulSoup) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for node in soup.select('script[type="application/ld+json"]'):
        raw_value = node.string or node.get_text(strip=True)
        if not raw_value:
            continue
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            continue
        _collect_schema_items(parsed, items)
    return [item for item in items if _looks_like_article_schema(item)]


def _collect_schema_items(value: Any, items: list[dict[str, Any]]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_schema_items(item, items)
        return
    if not isinstance(value, dict):
        return
    items.append(value)
    graph = value.get("@graph")
    if graph:
        _collect_schema_items(graph, items)


def _looks_like_article_schema(item: dict[str, Any]) -> bool:
    raw_type = item.get("@type")
    if isinstance(raw_type, str):
        types = {raw_type}
    elif isinstance(raw_type, list):
        types = {str(value) for value in raw_type}
    else:
        types = set()
    article_types = {"Article", "NewsArticle", "Report", "LiveBlogPosting", "BlogPosting"}
    return bool(types & article_types) or any(key in item for key in ("headline", "articleBody", "author", "publisher"))


def _schema_name_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(part for part in (_schema_name_value(item) for item in value) if part)
    if isinstance(value, dict):
        return _schema_text_value(value.get("name"))
    return ""


def _schema_text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for part in (_schema_text_value(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("text", "name", "headline", "description"):
            nested_value = _schema_text_value(value.get(key))
            if nested_value:
                return nested_value
    return ""


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for line in text.splitlines():
        cleaned = _clean_text(line)
        if len(cleaned.split()) >= 8:
            paragraphs.append(cleaned)
    if paragraphs:
        return paragraphs
    cleaned = _clean_text(text)
    return [cleaned] if cleaned else []


def _normalize_date_value(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    iso_candidate = cleaned.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).isoformat()
    except ValueError:
        return cleaned


def _extract_domain(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    return hostname.lower().removeprefix("www.")


def _clean_text(value: str) -> str:
    return " ".join((value or "").split())
