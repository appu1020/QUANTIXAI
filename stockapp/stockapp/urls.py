from django.contrib import admin
from django.urls import path, include
from myapp import api_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health', api_views.health_check_comprehensive, name='health_check_comprehensive'),
    path('', include('myapp.urls')),
]
