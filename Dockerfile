FROM python:3.11.4-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update ; \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    ; rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ; \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=8000 \
    GUNICORN_TIMEOUT=120 \
    WORKERS=4 \
    LOG_LEVEL=info

# Create necessary directories
RUN mkdir -p debug_images ; \
    chmod 777 debug_images

# Expose port (this is just documentation)
EXPOSE 8000

# Create a startup script
RUN echo '#!/bin/bash\n\
exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --timeout "${GUNICORN_TIMEOUT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}" \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    "app:create_app()"' > /app/start.sh ; \
    chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"] 