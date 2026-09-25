#!/bin/bash
set -e
D=<dataset_path>
MODEL=<qwen3.5-4b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
# Assumes the standard tuned Generate/SIS server is already running on
# localhost:30000. Uses --qps-per-worker 2 (5x more workers than the
# project default) because this model's own per-request service latency is
# high enough that the default qps/10 sizing becomes the dispatch
# bottleneck itself rather than the server -- see openloop_common.py and
# SUMMARY_QWEN3.5_4B.md Section 3's methodology note.
DUR=20
QPS_TARGETS="20 40 60 90 133 249"
for Q in $QPS_TARGETS; do
  echo "=== 4B Fused-Choice openloop qps=$Q ==="
  curl -s -X POST http://localhost:30000/flush_cache > /dev/null
  python benchmark_sglang_fused_choice_openloop.py --dataset-dir $D --qps $Q --duration $DUR \
    --qps-per-worker 2 --server http://localhost:30000 --output results_openloop_fused_choice_4b_qps$Q
done
echo FUSEDCHOICE_4B_OPENLOOP_DONE
