FROM python:3.11.4-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
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
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    GUNICORN_CMD_ARGS="--access-logfile=- --error-logfile=- --capture-output --enable-stdio-inheritance"

# Create necessary directories
RUN mkdir -p debug_images && \
    chmod 777 debug_images

# Expose port (this is just documentation)
EXPOSE 8000

# Run the application with proper error handling
CMD gunicorn \
    --bind "0.0.0.0:${PORT:-8000}" \
    --timeout 120 \
    --workers 4 \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    "app:create_app()" 