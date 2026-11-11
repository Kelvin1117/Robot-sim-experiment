#!/bin/bash
export PATH=/usr/bin:/bin
export DYGRO_ROOT=/common/users/cs2202/dygro_pi05_phase012_20261106
export DYGRO_PORT="$DYGRO_ROOT/oneshot_paper_v2"
export DYGRO_PYTHON=/common/home/cs2202/VLARLKit/model_backends/openpi/.venv/bin/python
export PYTHONPATH="${DYGRO_CODE:-$DYGRO_PORT}"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HOME="$DYGRO_ROOT/oneshot_openpi/cache/hf" XDG_CACHE_HOME="$DYGRO_ROOT/oneshot_openpi/cache"
export TMPDIR="$DYGRO_ROOT/oneshot_openpi/cache/tmp" TORCH_HOME="$DYGRO_ROOT/oneshot_openpi/cache/torch"
export TRITON_CACHE_DIR="$DYGRO_ROOT/oneshot_openpi/cache/triton" CUDA_CACHE_PATH="$DYGRO_ROOT/oneshot_openpi/cache/cuda"
export OPENPI_DATA_HOME=/common/users/cs2202/vlarlkit/cache/openpi
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export LIBERO_CONFIG_PATH=/common/users/cs2202/vlarlkit/config/libero
export NUMBA_CACHE_DIR="$DYGRO_ROOT/oneshot_openpi/cache/numba" MPLCONFIGDIR="$DYGRO_ROOT/oneshot_openpi/cache/matplotlib"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
