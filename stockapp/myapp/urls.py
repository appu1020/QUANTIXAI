from django.urls import path
from . import api_views, views

urlpatterns = [
    # ── Auth ──────────────────────────────────────────────────────────────
    path("",                              views.index,                name="index"),
    path("login",                         views.login,                name="login"),
    path("logout",                        views.logout,               name="logout"),
    path("register",                      views.register,             name="register"),

    # ── Pages ──────────────────────────────────────────────────────────────
    path("dashboard",                     views.dashboard,            name="dashboard"),
    path("portfolio",                     views.portfolio,            name="portfolio"),
    path("news",                          views.news,                 name="news"),
    path("market",                        views.market,               name="market"),
    path("add_to_wishlist/",              views.add_to_wishlist,      name="add_to_wishlist"),
    path("remove_from_wishlist/<str:symbol>/", views.remove_from_wishlist, name="remove_from_wishlist"),

    # ── REST API ───────────────────────────────────────────────────────────
    path("api/predict/",                  api_views.predict,               name="api_predict"),
    path("api/train/",                    api_views.train,                 name="api_train"),
    path("api/backtest/",                 api_views.backtest,              name="api_backtest"),
    path("api/live-data/",               api_views.live_data,             name="api_live_data"),
    path("api/price-stream/",            api_views.price_stream,          name="api_price_stream"),
    path("api/candle-data/",             api_views.candle_data,           name="api_candle_data"),
    path("api/sentiment/",               api_views.sentiment,             name="api_sentiment"),
    path("api/rag-query/",               api_views.rag_query,             name="api_rag_query"),
    path("api/model-performance/",       api_views.model_performance,     name="api_model_performance"),
    path("api/explain-prediction/",      api_views.explain_prediction,    name="api_explain_prediction"),
    path("api/invalidate-cache/",        api_views.invalidate_market_cache, name="api_invalidate_cache"),
    path("api/health/",                  api_views.health_check,          name="api_health"),
]
