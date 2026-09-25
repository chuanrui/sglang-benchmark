#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3.5-4b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
DUR=20
QPS_TARGETS="20 40 60 90 110 133"
for Q in $QPS_TARGETS; do
  echo "=== 4B N-calls Gen openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_jev_openloop.py --dataset-dir $D --qps $Q --duration $DUR \
    --workers-per-10-qps 2 --server http://localhost:30000 --output results_openloop_gen_ncalls_4b_qps$Q
done
for Q in $QPS_TARGETS; do
  echo "=== 4B Batched-Completions openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_batched_completions_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --workers-per-10-qps 2 --server http://localhost:30000 --output results_openloop_batched_completions_4b_qps$Q
done
for Q in $QPS_TARGETS; do
  echo "=== 4B SIS openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_score_api_openloop.py --dataset-dir $D --model-path $MODEL --qps $Q --duration $DUR \
    --flush-cache-interval 5 --workers-per-10-qps 2 --server http://localhost:30000 --output results_openloop_sis_4b_qps$Q
done
echo GENSIDE_4B_OPENLOOP_DONE
