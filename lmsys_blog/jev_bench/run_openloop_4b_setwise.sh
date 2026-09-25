#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3.5-4b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
# Assumes a setwise-configured server is already running on localhost:30000
# (--attention-backend flashinfer --disable-radix-cache
# --chunked-prefill-size -1, NO --enable-mis). Uses --qps-per-worker 2
# for the same reason as run_openloop_4b_fusedchoice.sh -- see that script's
# comment and SUMMARY_QWEN3.5_4B.md Section 3.
DUR=20
QPS_TARGETS="20 40 60 90 133 249"
for Q in $QPS_TARGETS; do
  echo "=== 4B Setwise openloop qps=$Q ==="
  python benchmark_sglang_setwise_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --qps-per-worker 2 --server http://localhost:30000 --output results_openloop_setwise_4b_qps$Q
done
echo SETWISE_4B_OPENLOOP_DONE
