#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/ecommerce-recsys}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

BACKEND_DIR="${BACKEND_DIR:-$PROJECT_DIR/mlflow}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_DIR/mlartifacts}"
BACKEND_DB="${BACKEND_DB:-$BACKEND_DIR/mlflow.db}"

SERVICE_NAME="${SERVICE_NAME:-mlflow}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
MLFLOW_BIN="$VENV_DIR/bin/mlflow"

echo "Creating directories..."
mkdir -p "$PROJECT_DIR"
mkdir -p "$BACKEND_DIR"
mkdir -p "$ARTIFACT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

echo "Upgrading pip..."
"$PIP_BIN" install --upgrade pip

echo "Installing MLflow..."
"$PIP_BIN" install mlflow

echo "Checking installation..."
"$MLFLOW_BIN" --version

ARTIFACT_URI="file://$ARTIFACT_DIR"
BACKEND_URI="sqlite:///$BACKEND_DB"

echo "Creating systemd service..."
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$MLFLOW_BIN server \
  --backend-store-uri $BACKEND_URI \
  --default-artifact-root $ARTIFACT_URI \
  --host $MLFLOW_HOST \
  --port $MLFLOW_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling service..."
sudo systemctl enable "$SERVICE_NAME"

echo "Starting service..."
sudo systemctl restart "$SERVICE_NAME"

echo "Service status:"
sudo systemctl --no-pager --full status "$SERVICE_NAME" || true

echo
echo "MLflow started."
echo "Backend URI:   $BACKEND_URI"
echo "Artifact URI:  $ARTIFACT_URI"
echo "Tracking URL:  http://$(hostname -I | awk '{print $1}'):$MLFLOW_PORT"
