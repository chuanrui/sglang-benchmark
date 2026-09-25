#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3-8b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
# Assumes a setwise-configured server is already running on localhost:30000
# (--attention-backend flashinfer --disable-radix-cache
# --chunked-prefill-size -1, NO --enable-mis). Cache-invariant by design, so
# no --flush-cache-interval flag exists on this script.
DUR=20
QPS_TARGETS="20 40 80 120 160 196"
for Q in $QPS_TARGETS; do
  echo "=== 8B Setwise openloop qps=$Q ==="
  python benchmark_sglang_setwise_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --server http://localhost:30000 --output results_openloop_setwise_8b_qps$Q
done
echo SETWISE_8B_OPENLOOP_DONE
