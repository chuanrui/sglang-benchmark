"""Open-loop (push-mode, Poisson-arrival) benchmark: batched Generate API
(one /v1/completions request per question).

Companion to benchmark_sglang_batched_completions.py, but replacing that
script's CLOSED-LOOP concurrency sweep with a genuine OPEN-LOOP / push-mode
load generator: requests arrive independently according to a Poisson process
at a caller-specified target QPS. See openloop_common.py's module docstring
for the full methodology and rationale.

Each question is still ONE /v1/completions request with all N candidate
prompts sent in a single native-batch `prompt: List[str]` list -- unchanged
from benchmark_sglang_batched_completions.py. Because /v1/completions does
not apply the chat template server-side (unlike /v1/chat/completions), this
script also applies it client-side via the tokenizer before sending each
candidate string, exactly as the closed-loop version does.

Usage:
    python benchmark_sglang_batched_completions_openloop.py \
        --dataset-dir <dataset_path> \
        --model-path <qwen3-8b_model_path> \
        --qps 40 --duration 60 --server http://localhost:30000 \
        --output results_openloop_batched_completions_qps40
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


def candidate_prompts(question):
    """Identical to benchmark_sglang_batched_completions.py's candidate_prompts()."""
    prefix = f"Context:\n{question['state']}\n\nQuestion: {question['question']}\n"
    return [
        prefix + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in question["options"]
    ]


def get_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def apply_chat_template(tokenizer, prompt):
    """Identical to benchmark_sglang_batched_completions.py's apply_chat_template()."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def call_sglang_batched(server, prompts, top_logprobs_n, timeout=60):
    """Identical to benchmark_sglang_batched_completions.py's call_sglang_batched()."""
    payload = json.dumps({
        "model": "default",
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": top_logprobs_n,
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        choices = body["choices"]
        if len(choices) != len(prompts):
            raise ValueError(f"expected {len(prompts)} choices, got {len(choices)}")
        by_index = {c["index"]: c for c in choices}
        if sorted(by_index.keys()) != list(range(len(prompts))):
            raise ValueError(f"choice indices not exactly 0..{len(prompts) - 1}: {sorted(by_index.keys())}")
        yes_logprobs = []
        for i in range(len(prompts)):
            choice = by_index[i]
            top = None
            lp = choice.get("logprobs")
            if lp and lp.get("top_logprobs"):
                top = lp["top_logprobs"][0]
            yes_lp = None
            if top:
                for tok, val in top.items():
                    if tok.strip().lower() == "yes":
                        yes_lp = val
                        break
            yes_logprobs.append(yes_lp)
        return {"success": True, "yes_logprobs": yes_logprobs,
                "prompt_tokens_total": body["usage"]["prompt_tokens"]}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError,
            ValueError, KeyError) as error:
        return {"success": False, "error_type": type(error).__name__, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True,
                         help="Local model path; used to load the tokenizer and apply its chat "
                              "template client-side (required -- /v1/completions does not apply it "
                              "server-side).")
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
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--qps-per-worker", type=int, default=10)
    parser.add_argument("--flush-cache-interval", type=float, default=5.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)

    print(f"Loading tokenizer from {args.model_path} to apply chat template client-side ...")
    tokenizer = get_tokenizer(args.model_path)
    prompts_by_question = [
        [apply_chat_template(tokenizer, p) for p in candidate_prompts(q)] for q in questions
    ]

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
        prompts = prompts_by_question[question_index]
        result = call_sglang_batched(args.server, prompts, args.top_logprobs)
        return {
            "success": result["success"],
            "num_candidates": len(prompts),
            "question_id": questions[question_index]["id"],
            "target_index": questions[question_index]["target_index"],
            "yes_logprobs": result.get("yes_logprobs"),
            "error": result.get("error"),
        }

    runner = OpenLoopRunner(args.qps, schedule, process_fn, qps_per_worker=args.qps_per_worker)
    records, wall_s = runner.run()

    if flush_thread is not None:
        stop_flush.set()
        flush_thread.join(timeout=5)

    with (output / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = 0
    graded = 0
    for r in records:
        lps = r.get("yes_logprobs")
        if r.get("success") and lps and all(lp is not None for lp in lps):
            predicted = max(range(len(lps)), key=lambda i: lps[i])
            correct += int(predicted == r["target_index"])
            graded += 1

    summary = summarize(records, wall_s, args.qps, extra={
        "approach": "generate-api-batched-completions-openloop",
        "server": args.server,
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions_pool": len(questions),
        "qps_per_worker": args.qps_per_worker,
        "num_workers": runner.num_workers,
        "flush_cache_interval": args.flush_cache_interval,
        "accuracy_proxy": {
            "description": "argmax P(yes) over candidates (from one batched /v1/completions call) "
                            "vs. dataset target, on questions where every candidate returned a yes "
                            "logprob",
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
