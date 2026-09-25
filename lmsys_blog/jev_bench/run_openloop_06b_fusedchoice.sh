#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3-0.6b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
# Assumes the standard tuned Generate/SIS server is already running on
# localhost:30000.
DUR=20
QPS_TARGETS="60 150 300 450 600 668"
for Q in $QPS_TARGETS; do
  echo "=== 0.6B Fused-Choice openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_fused_choice_openloop.py --dataset-dir $D --qps $Q --duration $DUR \
    --server http://localhost:30000 --output results_openloop_fused_choice_06b_qps$Q
done
echo FUSEDCHOICE_06B_OPENLOOP_DONE
