# Multi-stage Dockerfile for Modified Wallz full-stack application

# Stage 1: Build React TypeScript frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY web/frontend/package*.json ./
RUN npm ci

COPY web/frontend ./
RUN npm run build

# Stage 2: Python production runtime
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Copy compiled frontend distribution from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/web/frontend/dist

EXPOSE 5000

# Start unified Flask server
CMD ["python", "-m", "web.backend.app", "--port", "5000"]
