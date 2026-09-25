#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3-8b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
# Assumes the standard tuned Generate/SIS server is already running on
# localhost:30000 (--schedule-policy lpm --chunked-prefill-size 4096
# --enable-mixed-chunk --max-running-requests 256) -- Fused-Choice is an
# ordinary /v1/chat/completions call, needs no special server config.
DUR=20
QPS_TARGETS="20 40 80 120 160 196"
for Q in $QPS_TARGETS; do
  echo "=== 8B Fused-Choice openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_fused_choice_openloop.py --dataset-dir $D --qps $Q --duration $DUR \
    --server http://localhost:30000 --output results_openloop_fused_choice_8b_qps$Q
done
echo FUSEDCHOICE_8B_OPENLOOP_DONE
