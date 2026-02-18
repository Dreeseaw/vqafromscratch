# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# If you need git/build tools for some pip deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install your deps (kept separate for caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy repo
COPY . /app

# Safer default
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default: your local command
CMD ["python", "-m", "train.train_transformer", "lm_drop2"]

