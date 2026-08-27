# Vi-ViDoRe: Vietnamese Visual Document Retrieval Benchmark
# Docker image for reproducible experiments
# Build: docker build -t vi-vidore .
# Run: docker run --gpus all -it --rm -v ${PWD}:/workspace vi-vidore

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.lock /workspace/requirements.lock
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.lock

# Copy source code
COPY . /workspace/

# Install package in development mode
RUN python3 -m pip install -e . --no-deps

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8

# Default command
CMD ["bash"]