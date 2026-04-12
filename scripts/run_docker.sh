#!/bin/bash

set -e

IMAGE_NAME="ecommerce-recsys-api"
CONTAINER_NAME="ecommerce-recsys-container"
PORT=8000

echo "Stopping old container if exists..."
docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
docker rm $CONTAINER_NAME >/dev/null 2>&1 || true

echo "Building Docker image..."
docker build --no-cache -t $IMAGE_NAME .

echo "Starting container..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:8000 \
  $IMAGE_NAME

echo "Container started!"
echo "Swagger UI: http://localhost:$PORT/docs"
echo "Healthcheck: http://localhost:$PORT/health"
