from django.apps import AppConfig
from django.conf import settings
from pathlib import Path

class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'

    def ready(self):
        import sys
        if 'runserver' in sys.argv or 'gunicorn' in sys.modules or 'uvicorn' in sys.modules or 'waitress' in sys.modules:
            from myapp.services.model_manager import ModelManager
            MODEL_DIR = Path(settings.BASE_DIR) / "models"
            ModelManager.initialize(MODEL_DIR)
