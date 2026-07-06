#!/bin/bash

# Change directory to the Django project root
cd stockapp

# Run database migrations
echo "Running migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Start Gunicorn server
echo "Starting Gunicorn server..."
exec gunicorn stockapp.wsgi:application -c gunicorn_config.py
