import multiprocessing
import os

# Gunicorn configuration for Render (especially free/low tier)
# https://docs.gunicorn.org/en/stable/settings.html

bind = "0.0.0.0:" + os.environ.get("PORT", "8000")

# For ML apps, memory is a constraint. We use sync workers to keep memory bounded.
workers = int(os.environ.get("WEB_CONCURRENCY", 2))
# In some cases, we want to limit workers to 1 if memory is very constrained (Render Free has 512MB RAM).
if os.environ.get("RENDER"):
    workers = 1

# Timeout setting - higher than default (30) to allow for occasional slow ML inferences,
# though we've optimized dashboard loads to be async, prediction might still take a bit.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))

# Graceful restarts
max_requests = int(os.environ.get("MAX_REQUESTS", "100"))
max_requests_jitter = int(os.environ.get("MAX_REQUESTS_JITTER", "10"))

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app: load the application code before worker processes are forked.
# This saves memory but can cause issues if your app establishes DB connections on startup.
# We turn this on to save memory (copy-on-write).
preload_app = True
