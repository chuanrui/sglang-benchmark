"""Open-loop (push-mode, Poisson-arrival) benchmark: Score API (SIS/MIS).

Companion to benchmark_sglang_score_api.py, but replacing that script's
CLOSED-LOOP concurrency sweep with a genuine OPEN-LOOP / push-mode load
generator: requests arrive independently according to a Poisson process at a
caller-specified target QPS. See openloop_common.py's module docstring for
the full methodology and rationale.

Works unmodified against either a SIS server (no --enable-mis) or a MIS
server (--enable-mis) -- same request/response contract either way, exactly
like the closed-loop script. Pass --flush-cache-interval 0 when benchmarking
MIS (cache-invariant by design; flushing is a no-op there and unnecessary
overhead), and leave it at its default (>0) for SIS.

Usage:
    python benchmark_sglang_score_api_openloop.py \
        --dataset-dir <dataset_path> \
        --model-path <qwen3-8b_model_path> \
        --qps 40 --duration 60 --server http://localhost:30000 \
        --output results_openloop_sis_qps40
"""
import argparse
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from openloop_common import (
    OpenLoopRunner, build_poisson_schedule, flush_cache, load_questions,
    periodic_flush_thread, summarize,
)


def build_score_request(question):
    """Identical to benchmark_sglang_score_api.py's build_score_request()."""
    query = f"Context:\n{question['state']}\n\nQuestion: {question['question']}\n"
    items = [
        f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in question["options"]
    ]
    return query, items


def call_score_api(server, query, items, label_token_ids, timeout=60):
    """Identical to benchmark_sglang_score_api.py's call_score_api()."""
    payload = json.dumps({
        "model": "default",
        "query": query,
        "items": items,
        "label_token_ids": label_token_ids,
        "apply_softmax": True,
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/score", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return {"success": True, "scores": body["scores"], "prompt_tokens": body["usage"]["prompt_tokens"]}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        return {"success": False, "error_type": type(error).__name__, "error": str(error)}


def get_label_token_ids(model_path):
    """Identical to benchmark_sglang_score_api.py's get_label_token_ids()."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    return yes_id, no_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True, help="Local model path, to derive Yes/No label_token_ids")
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
    parser.add_argument("--workers-per-10-qps", type=int, default=10)
    parser.add_argument("--flush-cache-interval", type=float, default=5.0,
                         help="Set to 0 when benchmarking MIS (cache-invariant, flush is a no-op); "
                              "leave at the default (>0) for SIS.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    requests_by_question = [build_score_request(q) for q in questions]

    print(f"Deriving label token ids from {args.model_path} ...")
    yes_id, no_id = get_label_token_ids(args.model_path)
    print(f"label_token_ids: Yes={yes_id}, No={no_id}")

    schedule = build_poisson_schedule(args.qps, args.duration, len(questions), args.seed)
    print(f"Built Poisson schedule: {len(schedule)} requests over ~{args.duration}s at target qps={args.qps}")

    stop_flush = threading.Event()
    flush_thread = None
    if args.flush_cache_interval > 0:
        flush_cache(args.server)
        flush_thread = threading.Thread(
            target=periodic_flush_thread, args=(args.server, args.flush_cache_interval, stop_flush), daemon=True)
        flush_thread.start()

    def process_fn(question_index):
        query, items = requests_by_question[question_index]
        result = call_score_api(args.server, query, items, [yes_id, no_id])
        return {
            "success": result["success"],
            "num_candidates": len(items),
            "question_id": questions[question_index]["id"],
            "target_index": questions[question_index]["target_index"],
            "scores": result.get("scores"),
            "error": result.get("error"),
        }

    runner = OpenLoopRunner(args.qps, schedule, process_fn, workers_per_10_qps=args.workers_per_10_qps)
    records, wall_s = runner.run()

    if flush_thread is not None:
        stop_flush.set()
        flush_thread.join(timeout=5)

    with (output / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Accuracy proxy: argmax P(yes) over candidates (scores[i][0] = P(yes)
    # given apply_softmax=True), vs. dataset target.
    correct = 0
    graded = 0
    for r in records:
        scores = r.get("scores")
        if r.get("success") and scores:
            yes_probs = [s[0] for s in scores]
            predicted = max(range(len(yes_probs)), key=lambda i: yes_probs[i])
            correct += int(predicted == r["target_index"])
            graded += 1

    summary = summarize(records, wall_s, args.qps, extra={
        "approach": "score-api-openloop",
        "server": args.server,
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions_pool": len(questions),
        "workers_per_10_qps": args.workers_per_10_qps,
        "num_workers": runner.num_workers,
        "flush_cache_interval": args.flush_cache_interval,
        "accuracy_proxy": {
            "description": "argmax P(yes) over candidates (scores[i][0], apply_softmax=True) vs. "
                            "dataset target",
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
