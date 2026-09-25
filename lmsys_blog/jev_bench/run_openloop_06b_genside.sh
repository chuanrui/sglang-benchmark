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
  echo "=== 0.6B N-calls Gen openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_jev_openloop.py --dataset-dir $D --qps $Q --duration $DUR \
    --server http://localhost:30000 --output results_openloop_gen_ncalls_06b_qps$Q
done
for Q in $QPS_TARGETS; do
  echo "=== 0.6B Batched-Completions openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_batched_completions_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --server http://localhost:30000 --output results_openloop_batched_completions_06b_qps$Q
done
for Q in $QPS_TARGETS; do
  echo "=== 0.6B SIS openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_score_api_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --flush-cache-interval 5 --server http://localhost:30000 --output results_openloop_sis_06b_qps$Q
done
echo GENSIDE_06B_OPENLOOP_DONE
