#!/bin/bash
# Wrapper script to setup CUDA path compatibility and run inference

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python venv not found. Create it with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_DIR="$PROJECT_ROOT/venv/lib/$PY_VERSION/site-packages/nvidia"

if [ -d "$NVIDIA_DIR" ]; then
    NVIDIA_LIB_PATHS="$(find "$NVIDIA_DIR" -type d -name lib | tr '\n' ':')"
    export LD_LIBRARY_PATH="$PROJECT_ROOT/venv/compat_libs:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}"
fi

# Prevent PaddlePaddle from pre-allocating all GPU memory
export FLAGS_fraction_of_gpu_memory_to_use=0.2
export FLAGS_allocator_strategy=naive_best_fit

# Run inference script with all arguments passed to this shell script
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_inference.py" "$@"
