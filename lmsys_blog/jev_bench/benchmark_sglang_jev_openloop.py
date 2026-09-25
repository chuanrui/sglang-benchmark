"""Open-loop (push-mode, Poisson-arrival) benchmark: N-call Generate API.

Companion to benchmark_sglang_jev.py, but replacing that script's CLOSED-LOOP
concurrency sweep (fix N in-flight, measure wall time, repeat) with a genuine
OPEN-LOOP / push-mode load generator: requests arrive independently according
to a Poisson process at a caller-specified target QPS, exactly like real
production traffic, rather than being paced by how fast a fixed-size worker
pool can drain them. See openloop_common.py's module docstring for the full
methodology and why this matters (closed-loop sweeps cannot observe queueing
delay under overload -- "coordinated omission").

Each question still costs N separate /v1/chat/completions requests (one per
candidate option) -- unchanged from benchmark_sglang_jev.py. When a question's
scheduled arrival time comes up, its N candidate requests are fired together
via a small per-question thread pool (mirroring benchmark_sglang_jev.py's
--question-concurrency batched-dispatch mode) and we wait for the slowest to
return before considering that "job" (question) complete -- consistent with
every other approach in this project treating a question as done only once
all its candidates have been scored.

Usage:
    python benchmark_sglang_jev_openloop.py \
        --dataset-dir <dataset_path> \
        --qps 40 --duration 60 --server http://localhost:30000 \
        --output results_openloop_gen_qps40
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openloop_common import (
    OpenLoopRunner, build_poisson_schedule, flush_cache, load_questions,
    periodic_flush_thread, summarize,
)


def candidate_prompts(question):
    """Identical to benchmark_sglang_jev.py's candidate_prompts()."""
    prefix = f"Context:\n{question['state']}\n\nQuestion: {question['question']}\n"
    return [
        prefix + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in question["options"]
    ]


def call_sglang(server, prompt, timeout=60):
    """Identical to benchmark_sglang_jev.py's call_sglang()."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        choice = body["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("content", [{}])
        top = token_logprobs[0].get("top_logprobs", []) if token_logprobs else []
        yes_logprob = next((t["logprob"] for t in top if t["token"].strip().lower() == "yes"), None)
        return {"success": True, "yes_logprob": yes_logprob,
                "prompt_tokens": body["usage"]["prompt_tokens"]}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        return {"success": False, "error_type": type(error).__name__, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "calibration", "validation", "test", "ood"])
    parser.add_argument("--num-questions", type=int, default=100000,
                         help="Question pool size (loader clips to however many rows actually exist, "
                              "e.g. ~2408 in the test split). The schedule NEVER cycles/repeats a "
                              "question within one run (see build_poisson_schedule); if qps*duration "
                              "exceeds this pool size the run is capped and a warning is printed. "
                              "Default is large so the full available pool is used unless overridden.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qps", type=float, required=True, help="Target arrival rate (questions/sec).")
    parser.add_argument("--duration", type=float, required=True, help="Run duration in seconds.")
    parser.add_argument("--server", default="http://localhost:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers-per-10-qps", type=int, default=10,
                         help="Worker thread pool size = max(1, round(qps / this)). Default 10 matches "
                              "the project convention of qps/10 threads.")
    parser.add_argument("--flush-cache-interval", type=float, default=5.0,
                         help="Seconds between background /flush_cache calls during the run, "
                              "approximating the closed-loop scripts' --flush-cache-each-round cold-"
                              "cache methodology (there is no discrete round boundary here to hook a "
                              "flush onto). Set to 0 to disable (warm-cache run).")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    prompts_by_question = [candidate_prompts(q) for q in questions]

    schedule = build_poisson_schedule(args.qps, args.duration, len(questions), args.seed)
    print(f"Built Poisson schedule: {len(schedule)} requests over ~{args.duration}s at target qps={args.qps}")

    stop_flush = threading.Event()
    flush_thread = None
    if args.flush_cache_interval > 0:
        flush_cache(args.server)  # start cold
        flush_thread = threading.Thread(
            target=periodic_flush_thread, args=(args.server, args.flush_cache_interval, stop_flush), daemon=True)
        flush_thread.start()

    def process_fn(question_index):
        prompts = prompts_by_question[question_index]
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(call_sglang, args.server, p) for p in prompts]
            results = [f.result() for f in as_completed(futures)]
        yes_logprobs = [r.get("yes_logprob") for r in results]
        all_ok = all(r["success"] for r in results)
        return {
            "success": all_ok,
            "num_candidates": len(prompts),
            "question_id": questions[question_index]["id"],
            "target_index": questions[question_index]["target_index"],
            "yes_logprobs": yes_logprobs if all_ok else None,
            "errors": [r for r in results if not r["success"]] if not all_ok else None,
        }

    runner = OpenLoopRunner(args.qps, schedule, process_fn, workers_per_10_qps=args.workers_per_10_qps)
    records, wall_s = runner.run()

    if flush_thread is not None:
        stop_flush.set()
        flush_thread.join(timeout=5)

    with (output / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Accuracy proxy: argmax P(yes) over candidates vs. dataset target, only
    # on questions where every candidate succeeded (mirrors the other scripts).
    correct = 0
    graded = 0
    for r in records:
        if r.get("success") and r.get("yes_logprobs") and all(lp is not None for lp in r["yes_logprobs"]):
            predicted = max(range(len(r["yes_logprobs"])), key=lambda i: r["yes_logprobs"][i])
            correct += int(predicted == r["target_index"])
            graded += 1

    summary = summarize(records, wall_s, args.qps, extra={
        "approach": "generate-api-n-calls-openloop",
        "server": args.server,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions_pool": len(questions),
        "workers_per_10_qps": args.workers_per_10_qps,
        "num_workers": runner.num_workers,
        "flush_cache_interval": args.flush_cache_interval,
        "accuracy_proxy": {
            "description": "argmax P(yes) over candidates vs. dataset target, on questions where "
                            "every candidate succeeded",
            "correct": correct, "graded": graded,
            "accuracy": (correct / graded) if graded else None,
        },
    })
    with (output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(records)} samples to {output / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
