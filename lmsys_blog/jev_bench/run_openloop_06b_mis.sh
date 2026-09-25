#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3-0.6b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
DUR=20
QPS_TARGETS="60 150 300 450 600 668"
for Q in $QPS_TARGETS; do
  echo "=== 0.6B MIS openloop qps=$Q ==="
  python benchmark_sglang_score_api_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --flush-cache-interval 0 --server http://localhost:30000 --output results_openloop_mis_06b_qps$Q
done
echo MIS_06B_OPENLOOP_DONE
