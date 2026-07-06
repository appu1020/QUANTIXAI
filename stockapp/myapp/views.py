"""
views.py — Django template views for the QuantixAI web application.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import redirect, render

from .data_pipeline import download_market_data, fetch_candle_data, fetch_live_quotes_batch
from .model_engine import ModelLoader, load_metrics_report
from .models import Wishlist

logger = logging.getLogger(__name__)

MODEL_DIR = Path(settings.BASE_DIR) / "myapp" / "models"
TIME_STEPS = 60


# Model loaded via services


# ── Auth views ─────────────────────────────────────────────────────────────
def index(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "login.html")


def logout_view(request):
    auth_logout(request)
    return redirect("index")

# Keep old name for URL compatibility
logout = logout_view


def login(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect("dashboard")  # Fixed: was redirecting to undefined views
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, "login.html")


def register(request):
    if request.method == "POST":
        username  = request.POST.get("username", "").strip()
        email     = request.POST.get("email", "").strip()
        password1 = request.POST.get("password", "")
        password2 = request.POST.get("confirm_password", "")

        if not username:
            return render(request, "register.html", {"error": "Username is required."})
        if password1 != password2:
            return render(request, "register.html", {"error": "Passwords do not match."})
        if User.objects.filter(username=username).exists():
            return render(request, "register.html", {"error": "Username already taken."})

        try:
            User.objects.create_user(username=username, email=email, password=password1)
            messages.success(request, "Account created successfully! Please log in.")
            return redirect("login")
        except Exception as exc:
            logger.error("Registration failed: %s", exc)
            return render(request, "register.html", {"error": str(exc)})

    return render(request, "register.html")


# ── Dashboard ──────────────────────────────────────────────────────────────
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("login")

    symbol   = request.GET.get("symbol", "AAPL").upper().strip()
    interval = request.GET.get("interval", "1d").strip()

    context: dict = {
        "symbol":   symbol,
        "interval": interval,
        "intervals": ["5m", "15m", "1h", "1d"],
    }

    try:
        # Fast live quote for initial rendering
        from .data_pipeline import fetch_live_quote
        quote = fetch_live_quote(symbol)
        context["current_price"] = round(quote.get("close", 0.0), 4)
        context["price_change"] = round(quote.get("change", 0.0), 4)
        context["price_change_pct"] = round(quote.get("change_pct", 0.0), 2)
        
        # Load static model metrics
        from .services.model_manager import ModelManager
        from django.conf import settings
        from pathlib import Path
        import json
        
        metrics_path = Path(settings.BASE_DIR) / "myapp" / "models" / "model_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                context["model_metrics_json"] = f.read()

        # We NO LONGER run models synchronously.
        # Everything else (chart, prediction, sentiment, backtest) 
        # is handled by the async frontend API calls to prevent Render Gunicorn timeouts.
    except Exception as exc:
        logger.error("Dashboard error for %s: %s", symbol, exc)
        context["error"] = str(exc)

    return render(request, "index.html", context)


# ── Wishlist / Portfolio ───────────────────────────────────────────────────
@login_required
def add_to_wishlist(request):
    if request.method == "POST":
        symbol = request.POST.get("symbol", "").upper().strip()
        if not symbol:
            messages.error(request, "Invalid stock symbol.")
            return redirect("dashboard")
        if Wishlist.objects.filter(user=request.user, symbol=symbol).exists():
            messages.warning(request, f"{symbol} is already in your watchlist.")
        else:
            Wishlist.objects.create(user=request.user, symbol=symbol)
            messages.success(request, f"{symbol} added to watchlist.")
    return redirect("dashboard")


@login_required
def remove_from_wishlist(request, symbol):
    Wishlist.objects.filter(user=request.user, symbol=symbol.upper()).delete()
    messages.success(request, f"{symbol} removed from watchlist.")
    return redirect("portfolio")


def _get_stock_price(symbol: str) -> float | str:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            return "N/A"
        return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        return "N/A"


@login_required
def portfolio(request):
    """FIX M7: Batch-fetch all watchlist prices in a single yFinance call."""
    watchlist = Wishlist.objects.filter(user=request.user)
    symbols = [item.symbol for item in watchlist]
    stock_data = []

    if symbols:
        quotes = fetch_live_quotes_batch(symbols)
        for item in watchlist:
            q = quotes.get(item.symbol, {})
            price = q.get("close") or "N/A"
            change_pct = q.get("change_pct", 0.0)
            stock_data.append({
                "symbol":        item.symbol,
                "current_price": round(price, 2) if isinstance(price, float) and price > 0 else "N/A",
                "change_pct":    round(change_pct, 2) if isinstance(change_pct, float) else 0.0,
                "added_on":      item.added_on,
            })
    return render(request, "portfolio.html", {"watchlist": stock_data})


# ── News ───────────────────────────────────────────────────────────────────
def news(request):
    api_key = getattr(settings, "NEWSDATA_API_KEY", "")
    symbol  = request.GET.get("symbol", "stock market")
    news_data = []

    try:
        import requests as req
        if api_key:
            url = "https://newsdata.io/api/1/news"
            params = {"apikey": api_key, "q": symbol, "language": "en", "category": "business"}
            resp = req.get(url, params=params, timeout=10)
            resp.raise_for_status()
            news_data = resp.json().get("results", [])
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)


    # Fallback: Yahoo Finance Search API if no API key (Supports thumbnails)
    if not news_data:
        try:
            import requests as req
            from datetime import datetime, timezone
            
            ticker_symbol = request.GET.get("symbol", "SPY").upper()
            if ticker_symbol == "STOCK MARKET":
                ticker_symbol = "SPY"

            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f'https://query2.finance.yahoo.com/v1/finance/search?q={ticker_symbol}&newsCount=15'
            resp = req.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            raw_articles = resp.json().get('news', [])
            parsed = []

            for item in raw_articles:
                # Get the best thumbnail
                thumbnail = "/static/img/news_placeholder.svg"
                if "thumbnail" in item and "resolutions" in item["thumbnail"]:
                    resolutions = item["thumbnail"]["resolutions"]
                    if resolutions:
                        thumbnail = resolutions[0].get("url", thumbnail)
                
                # Parse date
                pub_date_str = ""
                if "providerPublishTime" in item:
                    try:
                        dt = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
                        pub_date_str = dt.strftime("%b %d, %Y")
                    except Exception:
                        pass

                parsed.append({
                    "title":       item.get("title", ""),
                    "link":        item.get("link", "#"),
                    "url":         item.get("link", "#"),
                    "source_id":   item.get("publisher", "Yahoo Finance"),
                    "publisher":   item.get("publisher", "Yahoo Finance"),
                    "description": item.get("description", "")[:300],
                    "pubDate":     pub_date_str,
                    "thumbnail":   thumbnail,
                })
            
            news_data = parsed
        except Exception as exc:
            logger.error("Yahoo Search API news parsing failed: %s", exc, exc_info=True)

    return render(request, "news.html", {"news_data": news_data, "symbol": symbol})


# ── Market overview ────────────────────────────────────────────────────────
def market(request):
    """Market overview using optimized batch fetching."""
    us_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "BRK-B", "V"]
    in_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                  "WIPRO.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS"]

    from .data_pipeline import fetch_live_quotes_batch
    
    def fetch_quotes(symbols: list[str]) -> list[dict]:
        results = []
        try:
            batch_data = fetch_live_quotes_batch(symbols)
            for sym in symbols:
                data = batch_data.get(sym, {})
                results.append({
                    "symbol":      sym.replace(".NS", ""),
                    "name":        sym.replace(".NS", ""), # Default to symbol if name not easily available
                    "price":       data.get("close", 0.0),
                    "change":      data.get("change", 0.0),
                    "change_pct":  data.get("change_pct", 0.0),
                    "volume":      data.get("volume", 0),
                })
        except Exception as exc:
            logger.error("Market fetch_quotes failed: %s", exc)
        return results

    us_stocks = fetch_quotes(us_symbols)
    indian_stocks = fetch_quotes(in_symbols)

    return render(request, "market.html", {
        "us_stocks":     us_stocks,
        "indian_stocks": indian_stocks,
    })
