# jev-bench: Open-Jev Serving Benchmarks for sglang

Benchmark scripts comparing five ways to serve a "score N candidate answers
for one question" workload on [sglang](https://github.com/sgl-project/sglang):

| Approach | Endpoint | Requests per question | Server flags required |
|---|---|---|---|
| **Generate API (N-calls)** | `/v1/chat/completions` | N (one per candidate) | none |
| **Generate API (Batched-Completions)** | `/v1/completions` | 1 | none |
| **Fused-Choice** | `/v1/chat/completions` | 1 (candidates inlined as lettered options) | none |
| **SIS** (Single-Item Score) | `/v1/score` | 1 (N independent `items`) | none |
| **MIS** (Multi-Item Score) | `/v1/score` | 1 (fused) | `--enable-mis` |
| **Setwise** | `/v1/score` | 1 (fused, single item + anchor tokens) | patched sglang build (see below) + `--disable-radix-cache --chunked-prefill-size -1` |

All approaches are benchmarked both **closed-loop** (fixed concurrency,
`benchmark_sglang_*.py`) and **open-loop** (genuine Poisson-arrival QPS,
`benchmark_sglang_*_openloop.py`). See each script's module docstring for
full methodology notes; the short version: closed-loop cannot observe
queueing delay under overload ("coordinated omission"), so the open-loop
scripts are the ones to trust for true matched-throughput comparisons.

## 1. Prerequisites

- A GPU host with Python 3.12 and CUDA.
- The [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev)
  dataset downloaded locally as parquet files. Scripts expect a directory
  (`--dataset-dir <dataset_path>`) containing parquet files readable by
  `pandas.read_parquet`, with at minimum these columns: `id`, `source`,
  `state_json` (JSON string), `question`, `options` (list of strings),
  `target` (list of floats/ints, argmax = correct option), `kind`
  (filtered to `"test"` by default via `--split`).
- A causal LM checkpoint compatible with sglang (this project used
  Qwen3-8B, Qwen3-0.6B, and Qwen3.5-4B; any HF-format causal LM should work
  for Generate/SIS/MIS/Fused-Choice — Setwise additionally requires the
  patched build below).
- `pip install sglang` (or build from source) inside a virtualenv
  (`<path_to_venv>` in the driver scripts below).

### Setwise-only: patched sglang build

CausalLM setwise scoring is not yet in stock sglang — it requires
[PR #41188](https://github.com/sgl-project/sglang/pull/41188)
("[Score API] Setwise scoring: CausalLM support (batched + `--enable-mis`)"),
an open PR against `sgl-project/sglang` that extends the (already-merged)
setwise-scoring primitive to CausalLM models.

To build it:
```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
gh pr checkout 41188   # or: git fetch https://github.com/sundar24295s/sglang.git suramach/setwise-causallm && git checkout FETCH_HEAD
pip install -e "python[all]"
```

## 2. Launching the server

All scripts talk to a running sglang server over HTTP (default
`http://localhost:30000`). Launch the server **before** running any
benchmark script, using the flags below for the approach you're testing.
`<model_path>` is any local path or HF repo id sglang accepts.

**Generate API (N-calls / Batched-Completions) and Fused-Choice** — no
special flags, just the project's tuned scheduling args:
```bash
python -m sglang.launch_server --model-path <model_path> --port 30000 \
  --schedule-policy lpm --chunked-prefill-size 4096 --enable-mixed-chunk \
  --max-running-requests 256
```

**SIS** — identical args to Generate API above (SIS's `/v1/score` with
independent `items` needs no special server config; keep the same tuned
args so the comparison is apples-to-apples).

**MIS**:
```bash
python -m sglang.launch_server --model-path <model_path> --port 30000 \
  --attention-backend flashinfer --enable-mis
```
(`--enable-mis` force-disables CUDA graph, radix cache, and chunked
prefill internally — no need to pass those separately.)

**Setwise** (patched build only):
```bash
python -m sglang.launch_server --model-path <model_path> --port 30000 \
  --disable-radix-cache --chunked-prefill-size -1
```
`--attention-backend flashinfer` is **not required** — the default backend
(`fa3` on Hopper) works identically; we verified this produces the same
throughput/latency and 0 errors. `--disable-radix-cache` and
`--chunked-prefill-size -1` **are** required: Setwise's anchor-token
pooling reads logits at absolute token positions from a single, complete
forward pass over the full prompt. Radix-cache reuse or chunked prefill
would each leave some anchor position's hidden state uncomputed for the
current request, and the server validates this at request time — you'll
get a clean `400 Bad Request` (not silently wrong results) if either flag
is missing:
```
score_extraction_token_id requires --disable-radix-cache because pooling
positions are relative to the full prompt.
```
We also confirmed (by deliberately patching out this validation) that
running Setwise with radix cache + chunked prefill enabled isn't just
"wrong results" — under load it hard-crashes the CUDA context with an
out-of-bounds `vectorized_gather_kernel` assertion. Don't disable this
guard in production.

## 3. Running the benchmarks

### Closed-loop (concurrency sweep)

```bash
source <path_to_venv>/bin/activate
python benchmark_sglang_jev.py \
  --dataset-dir <dataset_path> \
  --split test --num-questions 50 --warmup 3 --repetitions 10 \
  --server http://localhost:30000 --output results --concurrency 1
```
Each `benchmark_sglang_*.py` script has the same overall shape (see its
module docstring for the full flag list and a worked `Usage:` example),
but not every script needs `--model-path`: it's required only by the
scripts that must locally tokenize label/anchor tokens (SIS, Setwise) or
apply a chat template client-side (Batched-Completions) — plain Generate
API and Fused-Choice only talk to the server and don't need it. Sweep
concurrency by running the script once per concurrency level with
`--concurrency N`.

### Open-loop (true Poisson-arrival QPS) — recommended

```bash
python benchmark_sglang_setwise_openloop.py \
  --dataset-dir <dataset_path> --model-path <model_path> \
  --qps 40 --duration 60 --server http://localhost:30000 \
  --output results_openloop_setwise_qps40
```
Key flags (see `openloop_common.py`'s module docstring for the full
rationale):
- `--qps`: target requests/second (Poisson arrivals).
- `--duration`: wall-clock seconds to run.
- `--workers-per-10-qps`: dispatch worker pool size, default assumes
  ~100ms/request (i.e. `qps/10` workers). **Increase this** (e.g. to `2`,
  giving 5x more workers) for slower models where per-request latency
  approaches or exceeds 100ms — otherwise the worker pool itself becomes
  the bottleneck, not the server, and you'll silently under-measure the
  server's real ceiling. Sanity-check via `num_workers` and
  `service_latency_ms` in the output `summary.json`.
- The Poisson schedule **never repeats a question** within one run (raises
  `--num-questions` as needed, capped to the dataset's pool size) —
  critical so no two requests in a single run share identical prompt
  text, which would otherwise contaminate results with prefix-cache hits
  at high QPS.

### Driver scripts (full sweeps)

`run_openloop_{8b,06b,4b}_{genside,mis,fusedchoice,setwise}.sh` run a full
QPS sweep for one model/approach in one shot. Before running one, fill in
the placeholders at the top of the script:
```bash
D=<dataset_path>
MODEL=<qwen3-8b_model_path>          # or <qwen3-0.6b_model_path> / <qwen3.5-4b_model_path>
source <path_to_venv>/bin/activate
export LD_LIBRARY_PATH=<path_to_venv>/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
```
Then launch the matching server config from Section 2 on `localhost:30000`
and run the script (e.g. `bash run_openloop_8b_setwise.sh`). Each driver
prints a `..._DONE` sentinel on success and writes one
`results_openloop_<approach>_<model>_qps<N>/summary.json` per QPS target.

`run_openloop_{06b,4b}_lowqps_*.sh` are supplementary drivers that only
cover a model's lowest QPS targets — useful for filling in resolution at
the low end without rerunning the full sweep.

## 4. Output format

Every script writes `results/summary.json` (or wherever `--output` points)
containing: `achieved_qps` (open-loop only), `end_to_end_latency_ms` /
`service_latency_ms` percentiles (p50/p95/p99/mean/stdev),
`*_by_candidate_count` breakdowns, `num_errors`, and an `accuracy_proxy`
(argmax-vs-target agreement — a rough sanity check, not a rigorous
accuracy metric). Raw per-request samples are written to
`<output>/samples.jsonl`.
