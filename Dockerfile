FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# --- Base env ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Unset LD_LIBRARY_PATH to prevent conflicts with JAX CUDA libraries
ENV LD_LIBRARY_PATH=""

# Install system dependencies
RUN apt update && apt install -y \
    curl \
    python3 \
    python3-pip \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip
RUN pip3 install uv

# Add uv to PATH and verify installation
ENV PATH="/usr/local/bin:$PATH"
RUN which uv && uv --version

# --- Application setup ---
WORKDIR /app

# ---- deps layer (cache) ----
COPY pyproject.toml uv.lock* /app/
RUN uv sync --extra cuda --extra dev --frozen

# ---- code layer ----
COPY . /app/
RUN uv pip install -e .
