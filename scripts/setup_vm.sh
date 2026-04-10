#!/bin/bash
set -e

echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    htop \
    tmux \
    unzip \
    build-essential \
    pkg-config \
    libgomp1

echo "Creating project folders..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p models
mkdir -p mlruns
mkdir -p logs
mkdir -p scripts
mkdir -p docker
mkdir -p airflow/dags
mkdir -p src/api
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/monitoring
mkdir -p src/utils

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip tools..."
pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping dependency installation."
fi

echo "Setup completed successfully."
echo "Activate environment with:"
echo "source .venv/bin/activate"
