"""
sentiment_engine.py — Market News Sentiment Analysis Engine.

Data sources:
  1. yfinance Ticker.news (free, no API key required)
  2. NewsData.io API (requires NEWSDATA_API_KEY)
  3. Reddit PRAW (optional, requires REDDIT_* env vars)

Sentiment scoring uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
which is tuned for financial/social media text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class ArticleSentiment:
    title: str
    source: str
    url: str
    published_at: str
    sentiment_label: str    # BULLISH | BEARISH | NEUTRAL
    compound_score: float   # -1 to +1
    bullish_score: float
    bearish_score: float


@dataclass
class SentimentScore:
    symbol: str
    bullish_score: float
    bearish_score: float
    impact_score: float       # net score: bullish - bearish
    article_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    summary: str
    articles: list[ArticleSentiment] = field(default_factory=list)


def analyze_text(text: str) -> dict[str, float]:
    """Run VADER sentiment analysis on text. Returns pos/neg/neu/compound scores."""
    scores = _analyzer.polarity_scores(text)
    return {
        "positive": float(scores["pos"] * 100),
        "negative": float(scores["neg"] * 100),
        "neutral":  float(scores["neu"] * 100),
        "compound": float(scores["compound"]),
    }


def score_article(title: str, description: str, source: str, url: str, published_at: str) -> ArticleSentiment:
    """Score a single news article."""
    text = f"{title}. {description}".strip()
    scores = analyze_text(text)

    compound = scores["compound"]
    bullish = min(100.0, float(scores["positive"] * (1 + max(compound, 0))))
    bearish = min(100.0, float(scores["negative"] * (1 - min(compound, 0))))

    if compound >= 0.05:
        label = "BULLISH"
    elif compound <= -0.05:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return ArticleSentiment(
        title=title[:200],
        source=source,
        url=url,
        published_at=published_at,
        sentiment_label=label,
        compound_score=round(compound, 4),
        bullish_score=round(bullish, 2),
        bearish_score=round(bearish, 2),
    )


# ── Data fetchers ──────────────────────────────────────────────────────────

def fetch_yahoo_news(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetch news from Yahoo Finance RSS feed.
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US'
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.text)
        articles = []
        for item in root.findall('./channel/item')[:limit]:
            title = item.findtext('title', '')
            description = item.findtext('description', '')
            link = item.findtext('link', '')
            pub_date = item.findtext('pubDate', '')
            
            articles.append({
                "title":        title,
                "description":  description,
                "source":       "Yahoo Finance",
                "url":          link,
                "published_at": pub_date,
            })
            
        logger.info("Fetched %d Yahoo Finance news for %s", len(articles), symbol)
        return articles
    except Exception as exc:
        logger.warning("Yahoo Finance news fetch failed for %s: %s", symbol, exc)
        return []


def fetch_newsdata_api(symbol: str, api_key: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch news from NewsData.io API."""
    try:
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey":   api_key,
            "q":        symbol,
            "language": "en",
            "category": "business",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])[:limit]
        return [
            {
                "title":        item.get("title", ""),
                "description":  item.get("description", ""),
                "source":       item.get("source_id", "newsdata.io"),
                "url":          item.get("link", ""),
                "published_at": item.get("pubDate", ""),
            }
            for item in results
        ]
    except Exception as exc:
        logger.warning("NewsData.io fetch failed: %s", exc)
        return []


def fetch_reddit_news(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch Reddit posts mentioning the symbol from r/stocks and r/investing."""
    try:
        import praw
        from django.conf import settings
        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
        )
        subreddits = ["stocks", "investing", "wallstreetbets"]
        articles = []
        for sub in subreddits:
            for post in reddit.subreddit(sub).search(symbol, limit=limit // len(subreddits)):
                articles.append({
                    "title":        post.title,
                    "description":  post.selftext[:300],
                    "source":       f"reddit/r/{sub}",
                    "url":          f"https://reddit.com{post.permalink}",
                    "published_at": datetime.fromtimestamp(
                        post.created_utc, tz=timezone.utc
                    ).isoformat(),
                })
        logger.info("Fetched %d Reddit posts for %s", len(articles), symbol)
        return articles
    except ImportError:
        logger.debug("praw not installed — skipping Reddit news.")
        return []
    except Exception as exc:
        logger.warning("Reddit fetch failed for %s: %s", symbol, exc)
        return []


def fetch_news_for_symbol(symbol: str, api_key: str | None = None, include_reddit: bool = False) -> list[dict[str, Any]]:
    """
    Aggregate news from all available sources.

    Order of preference:
    1. Yahoo Finance (always available)
    2. NewsData.io (if api_key configured)
    3. Reddit PRAW (if include_reddit=True and credentials configured)
    """
    articles: list[dict[str, Any]] = []
    articles.extend(fetch_yahoo_news(symbol, limit=8))

    if api_key:
        articles.extend(fetch_newsdata_api(symbol, api_key=api_key, limit=5))

    if include_reddit:
        articles.extend(fetch_reddit_news(symbol, limit=5))

    # Deduplicate by URL and normalized title (FIX N1)
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        title = a.get("title", "")
        title_key = title.lower().strip()[:80] if title else ""

        is_duplicate = False
        if url and url in seen_urls:
            is_duplicate = True
        if title_key and title_key in seen_titles:
            is_duplicate = True

        if not is_duplicate:
            if url:
                seen_urls.add(url)
            if title_key:
                seen_titles.add(title_key)
            unique.append(a)

    return unique


def aggregate_sentiment(
    symbol: str,
    articles: list[dict[str, Any]] | None = None,
    texts: list[str] | None = None,
) -> SentimentScore:
    """
    Compute aggregate sentiment from a list of articles or raw texts.

    Prefers structured `articles` list (with title, description, source, url, published_at)
    over raw `texts`.
    """
    scored_articles: list[ArticleSentiment] = []

    if articles:
        for a in articles:
            title = a.get("title", "")
            if not title:
                continue
            scored = score_article(
                title=title,
                description=a.get("description", ""),
                source=a.get("source", ""),
                url=a.get("url", ""),
                published_at=a.get("published_at", ""),
            )
            scored_articles.append(scored)
    elif texts:
        for i, text in enumerate(texts):
            if not text:
                continue
            scored = score_article(
                title=text[:100],
                description=text,
                source="unknown",
                url="",
                published_at="",
            )
            scored_articles.append(scored)

    if not scored_articles:
        return SentimentScore(
            symbol=symbol,
            bullish_score=0.0, bearish_score=0.0, impact_score=0.0,
            article_count=0, bullish_count=0, bearish_count=0, neutral_count=0,
            summary="No news data available for sentiment analysis.",
        )

    bullish_sum = sum(a.bullish_score for a in scored_articles)
    bearish_sum = sum(a.bearish_score for a in scored_articles)
    n = len(scored_articles)
    bullish_avg = bullish_sum / n
    bearish_avg = bearish_sum / n
    impact = round(bullish_avg - bearish_avg, 2)

    bullish_count = sum(1 for a in scored_articles if a.sentiment_label == "BULLISH")
    bearish_count = sum(1 for a in scored_articles if a.sentiment_label == "BEARISH")
    neutral_count = sum(1 for a in scored_articles if a.sentiment_label == "NEUTRAL")

    if bullish_avg > bearish_avg + 5:
        summary = f"Positive market sentiment: {bullish_count}/{n} articles bullish."
    elif bearish_avg > bullish_avg + 5:
        summary = f"Negative market sentiment: {bearish_count}/{n} articles bearish."
    else:
        summary = f"Mixed or neutral sentiment across {n} articles."

    # Cache to DB
    _cache_articles_to_db(symbol, scored_articles)

    return SentimentScore(
        symbol=symbol,
        bullish_score=round(bullish_avg, 2),
        bearish_score=round(bearish_avg, 2),
        impact_score=impact,
        article_count=n,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        summary=summary,
        articles=scored_articles,
    )


def _cache_articles_to_db(symbol: str, articles: list[ArticleSentiment]) -> None:
    """Persist scored articles to NewsArticle DB model (best-effort)."""
    try:
        from .models import NewsArticle
        for a in articles:
            if not a.url:
                continue
            NewsArticle.objects.update_or_create(
                source=a.source,
                url=a.url,
                defaults={
                    "symbol":        symbol,
                    "title":         a.title,
                    "sentiment":     a.compound_score,
                    "bullish_score": a.bullish_score,
                    "bearish_score": a.bearish_score,
                    "impact_score":  a.bullish_score - a.bearish_score,
                },
            )
    except Exception as exc:
        logger.debug("Could not cache articles to DB: %s", exc)
