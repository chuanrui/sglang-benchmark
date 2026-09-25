"""Open-loop (push-mode, Poisson-arrival) benchmark: Fused-Choice Generate API.

Companion to benchmark_sglang_fused_choice.py, but replacing that script's
CLOSED-LOOP concurrency sweep with a genuine OPEN-LOOP / push-mode load
generator: requests arrive independently according to a Poisson process at a
caller-specified target QPS. See openloop_common.py's module docstring for
the full methodology and rationale (queueing delay under overload cannot be
observed by a closed-loop concurrency sweep -- "coordinated omission").

Each question is still ONE /v1/chat/completions request with all of a
question's candidates inlined as lettered options (A, B, C, ...) -- unchanged
from benchmark_sglang_fused_choice.py. The model generates a single output
token (max_tokens=1); rather than trusting the sampled token, we read the
logprobs of every candidate LETTER at that position from `top_logprobs` and
take the argmax, exactly like the closed-loop script.

No tokenizer / model-path needed: unlike the Score API family, the Generate
API returns each top_logprobs entry as decoded text, so candidate letters are
matched by string comparison, not token id.

Usage:
    python benchmark_sglang_fused_choice_openloop.py \
        --dataset-dir <dataset_path> \
        --qps 40 --duration 60 --server http://localhost:30000 \
        --output results_openloop_fused_choice_qps40
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

LETTERS = "ABCDEFGHIJKLMNOP"  # supports up to 16 options, this dataset's max


def build_fused_prompt(question):
    """Identical to benchmark_sglang_fused_choice.py's build_fused_prompt()."""
    n = len(question["options"])
    letters = LETTERS[:n]
    prefix = (
        f"Context:\n{question['state']}\n\n"
        f"Question: {question['question']}\n\n"
        f"Options:\n"
    )
    lines = "\n".join(f"{letters[i]}) {option}" for i, option in enumerate(question["options"]))
    letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}" if n > 1 else letters
    suffix = (
        f"\n\nChoose the single correct option above. "
        f"Your entire response must be exactly one letter: {letter_list}. "
        f"Do not include any other words, punctuation, or explanation."
    )
    return prefix + lines + suffix


def extract_letter_logprobs(top_logprobs, letters):
    """Identical to benchmark_sglang_fused_choice.py's extract_letter_logprobs()."""
    result = {letter: None for letter in letters}
    for entry in top_logprobs:
        text = entry.get("token", "").strip().upper()
        if text in result and result[text] is None:
            result[text] = entry.get("logprob")
    return result


def call_sglang_fused(server, prompt, letters, top_logprobs_n, timeout=60):
    """Identical to benchmark_sglang_fused_choice.py's call_sglang_fused() (latency
    timing removed here -- OpenLoopRunner measures dispatch/completion itself)."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs_n,
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
        letter_logprobs = extract_letter_logprobs(top, letters)
        return {
            "success": True,
            "sampled_token": choice["message"]["content"],
            "letter_logprobs": letter_logprobs,
            "prompt_tokens": body["usage"]["prompt_tokens"],
        }
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
    parser.add_argument("--top-logprobs", type=int, default=20,
                         help="Must be >= the max candidate count in the sample (this dataset maxes "
                              "at 16); set with margin so the correct letter is very likely captured "
                              "even when the model doesn't rank it in the top few tokens.")
    parser.add_argument("--qps-per-worker", type=int, default=10,
                         help="Worker thread pool size = max(1, round(qps / this)). Default 10 matches "
                              "the project convention of qps/10 threads; use a smaller value (e.g. 2) "
                              "for heavier models whose own per-request latency approaches or exceeds "
                              "~100ms, so the worker pool itself doesn't become the dispatch "
                              "bottleneck -- see openloop_common.py's module docstring.")
    parser.add_argument("--flush-cache-interval", type=float, default=5.0,
                         help="Seconds between background /flush_cache calls during the run, "
                              "approximating the closed-loop script's --flush-cache-each-round cold-"
                              "cache methodology. Set to 0 to disable (warm-cache run). Not load-"
                              "bearing for correctness -- the schedule's no-repeat guarantee is what "
                              "actually prevents cache contamination; see openloop_common.py.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    prompts_by_question = [
        (build_fused_prompt(q), LETTERS[:len(q["options"])]) for q in questions
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
        prompt, letters = prompts_by_question[question_index]
        result = call_sglang_fused(args.server, prompt, letters, args.top_logprobs)
        return {
            "success": result["success"],
            "num_candidates": len(letters),
            "question_id": questions[question_index]["id"],
            "target_index": questions[question_index]["target_index"],
            "letter_logprobs": result.get("letter_logprobs"),
            "sampled_token": result.get("sampled_token"),
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

    # Accuracy proxy: argmax over candidate-LETTER logprobs (not the sampled
    # token) vs. dataset target, only on questions where every candidate
    # letter's logprob was captured within top_logprobs.
    correct = 0
    graded = 0
    ungraded_missing_logprob = 0
    for r in records:
        lp = r.get("letter_logprobs")
        if not (r.get("success") and lp):
            continue
        if any(v is None for v in lp.values()):
            ungraded_missing_logprob += 1
            continue
        letters_sorted = sorted(lp.keys())
        predicted_letter = max(letters_sorted, key=lambda letter: lp[letter])
        predicted_index = LETTERS.index(predicted_letter)
        correct += int(predicted_index == r["target_index"])
        graded += 1

    summary = summarize(records, wall_s, args.qps, extra={
        "approach": "fused-choice-openloop",
        "server": args.server,
        "top_logprobs": args.top_logprobs,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions_pool": len(questions),
        "qps_per_worker": args.qps_per_worker,
        "num_workers": runner.num_workers,
        "flush_cache_interval": args.flush_cache_interval,
        "accuracy_proxy": {
            "description": "argmax over candidate-LETTER logprobs (not the greedily sampled token) "
                            "vs. dataset target, on questions where every candidate letter's logprob "
                            "was found within top_logprobs.",
            "correct": correct, "graded": graded,
            "accuracy": (correct / graded) if graded else None,
            "ungraded_missing_a_letter_logprob": ungraded_missing_logprob,
        },
    })
    with (output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(records)} samples to {output / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
