# Stage 1: Build & Python Dependency Installation
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

# Set shell and non-interactive frontend
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Python and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements file first to utilize Docker layer caching
COPY requirements.txt .

# Install dependencies using pip (and target native CPU/GPU support)
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Final Production Runner Stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime system packages (no build-essential to keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root system user for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Create necessary directories and set ownership before copying files
RUN mkdir -p data/uploads outputs/results config logs && \
    chown -R appuser:appgroup /workspace

# Copy source code and config files
COPY --chown=appuser:appgroup app /workspace/app
COPY --chown=appuser:appgroup config /workspace/config
COPY --chown=appuser:appgroup scripts /workspace/scripts
COPY --chown=appuser:appgroup pyproject.toml /workspace/
COPY --chown=appuser:appgroup README.md /workspace/

# Switch to the non-root user
USER appuser

# Install package in editable mode to expose entrypoint CLI tools
RUN pip3 install --no-cache-dir --user -e .

# Expose port for FastAPI API
EXPOSE 8000

# Set environment variables for CUDA and GPU library paths
ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV FLAGS_fraction_of_gpu_memory_to_use=0.2
ENV FLAGS_allocator_strategy=naive_best_fit

# Start the FastAPI server using the packaged entrypoint CLI command
CMD ["smart-gate-api"]
