"""Benchmark sglang (Qwen3-8B) single-output-token latency on the Open-Jev dataset.

Mirrors the request shape used by Zefan-Cai/Open-Jev's own
scripts/benchmark_inference_latency.py + jev/api.py candidate_prompts():
for every typed-decision "question", each candidate answer is rendered as an
independent yes/no prompt ("Proposed answer: <option>\\nIs this proposed
answer correct? Answer Yes or No.") and scored with a single generated token
(max_tokens=1). Unlike Open-Jev's fine-tuned classifier head, this drives a
generic instruct model (Qwen3-8B) served by sglang over its OpenAI-compatible
HTTP API, with thinking mode disabled so the first token is the answer.

Usage:
    python benchmark_sglang_jev.py \
        --dataset-dir <dataset_path> \
        --split test --num-questions 50 --warmup 3 --repetitions 10 \
        --server http://localhost:30000 --output results
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def percentile(values, q):
    """Linear interpolation percentile (matches numpy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_questions(dataset_dir, split, num_questions, seed, kind="choice"):
    files = sorted(glob.glob(os.path.join(dataset_dir, f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split!r} under {dataset_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["kind"] == kind]
    if df.empty:
        raise ValueError(f"No rows with kind={kind!r} in split={split!r}")
    sample = df.sample(n=min(num_questions, len(df)), random_state=seed).to_dict(orient="records")
    questions = []
    for row in sample:
        state = json.loads(row["state_json"])
        options = list(row["options"])
        target = list(row["target"])
        target_index = target.index(max(target))
        questions.append({
            "id": row["id"],
            "source": row["source"],
            "state": state,
            "question": row["question"],
            "options": options,
            "target_index": target_index,
        })
    return questions


def candidate_prompts(question):
    """Render each candidate option as an isolated yes/no prompt (Open-Jev jev/api.py)."""
    prefix = f"Context:\n{question['state']}\n\nQuestion: {question['question']}\n"
    return [
        prefix + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in question["options"]
    ]


def call_sglang(server, prompt, timeout=60):
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
        # Qwen3 defaults to emitting a <think> block first; disable so the
        # single generated token is the yes/no answer itself.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{server}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    begin = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        end = time.perf_counter()
        wall_ms = (end - begin) * 1000
        choice = body["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("content", [{}])
        top = token_logprobs[0].get("top_logprobs", []) if token_logprobs else []
        yes_logprob = next((t["logprob"] for t in top if t["token"].strip().lower() == "yes"), None)
        return {
            "success": True,
            "wall_ms": wall_ms,
            "start_ts": begin,
            "end_ts": end,
            "token": choice["message"]["content"],
            "yes_logprob": yes_logprob,
            "prompt_tokens": body["usage"]["prompt_tokens"],
            "completion_tokens": body["usage"]["completion_tokens"],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        end = time.perf_counter()
        return {"success": False, "wall_ms": (end - begin) * 1000,
                "start_ts": begin, "end_ts": end,
                "error_type": type(error).__name__, "error": str(error)}


def flush_cache(server, timeout=30):
    """POST /flush_cache -- clears sglang's RadixAttention prefix-cache tree.

    Only succeeds when no requests are running/waiting (safe to call between
    rounds, since we always wait for the full round to finish first). Used to
    force every round to be a genuinely cold-cache measurement, removing the
    cross-round prefix-reuse benefit from replaying the identical 200
    questions every round (see --flush-cache-each-round).
    """
    req = urllib.request.Request(f"{server}/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True,
                         help="Directory containing <split>-*.parquet (e.g. .../data/release-v2-redistributable)")
    parser.add_argument("--split", default="test", choices=["train", "calibration", "validation", "test", "ood"])
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--server", default="http://localhost:30000")
    parser.add_argument("--output", required=True, help="Output directory for samples.jsonl + summary.json")
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Number of in-flight requests, at CANDIDATE granularity. 1 = original "
                              "sequential single-stream latency test. >1 fires all (question, "
                              "candidate) requests for a round through one flat thread pool of this "
                              "size -- candidates from different questions are interleaved arbitrarily "
                              "(the pool doesn't know which question a request belongs to). Ignored "
                              "when --question-concurrency is set.")
    parser.add_argument("--question-concurrency", type=int, default=0,
                         help="Number of in-flight requests, at QUESTION granularity (the realistic "
                              "client dispatch pattern: a caller receives one question, knows its N "
                              "candidates, and fires all N together). This many questions are "
                              "processed concurrently via an outer ThreadPoolExecutor(max_workers=N); "
                              "each of those in-flight questions independently submits all of its own "
                              "candidates to its own inner ThreadPoolExecutor and awaits them together "
                              "before that outer worker picks up the next question. "
                              "--question-concurrency 1 reproduces the old --per-question-concurrency "
                              "flag's behavior (one question in flight at a time, low-load, comparable "
                              "to SIS/MIS's concurrency=1); --question-concurrency N>1 sweeps this same "
                              "batched-dispatch pattern to higher load, unlike the old flag which had no "
                              "sweep capability. Overrides --concurrency when >0.")
    parser.add_argument("--flush-cache-each-round", action="store_true",
                         help="POST /flush_cache before every round (including warmup rounds), so "
                              "every round is a genuinely cold radix-cache measurement -- removes the "
                              "cross-round prefix-reuse benefit from replaying the identical question "
                              "set every round. Without this flag, only the first couple of warmup "
                              "rounds pay cold-cache cost; all later rounds (including all 'measured' "
                              "ones) run on an already-warm cache.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    print(f"Loaded {len(questions)} questions "
          f"({sum(len(q['options']) for q in questions)} total candidates), "
          f"concurrency={'question=' + str(args.question_concurrency) if args.question_concurrency else args.concurrency}")

    # Flatten to one task per (question, candidate); every round (warmup or measured)
    # re-issues the full task list, optionally in parallel via the thread pool.
    tasks = []
    tasks_by_question = []
    for qi, question in enumerate(questions):
        group = [(qi, ci, prompt) for ci, prompt in enumerate(candidate_prompts(question))]
        tasks_by_question.append(group)
        tasks.extend(group)

    yes_logprobs = {qi: [None] * len(q["options"]) for qi, q in enumerate(questions)}
    samples = []
    write_lock = threading.Lock()
    round_stats = []

    with (output / "samples.jsonl").open("w") as raw_file:
        def run_task(qi, ci, prompt, phase, repetition):
            result = call_sglang(args.server, prompt)
            sample = {
                "question_id": questions[qi]["id"], "source": questions[qi]["source"],
                "candidate_index": ci, "num_candidates": len(questions[qi]["options"]),
                "phase": phase, "repetition": repetition, **result,
            }
            with write_lock:
                samples.append(sample)
                raw_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                raw_file.flush()
            if phase == "measured" and result["success"]:
                yes_logprobs[qi][ci] = result.get("yes_logprob")
            return sample

        for iteration in range(args.warmup + args.repetitions):
            phase = "warmup" if iteration < args.warmup else "measured"
            repetition = iteration if phase == "warmup" else iteration - args.warmup
            if args.flush_cache_each_round:
                flush_cache(args.server)
                time.sleep(1)
            round_start = time.perf_counter()
            if args.question_concurrency:
                # Question-granularity concurrency: N questions in flight at once (outer
                # pool), each independently batch-dispatching its own candidates (inner
                # pool) and awaiting them before that outer worker moves to the next
                # question. --question-concurrency 1 is the old --per-question-concurrency
                # behavior (one question in flight, low-load); N>1 sweeps the same
                # batched-dispatch pattern to higher load.
                def process_question(group):
                    with ThreadPoolExecutor(max_workers=len(group)) as inner_pool:
                        futures = [inner_pool.submit(run_task, qi, ci, prompt, phase, repetition) for qi, ci, prompt in group]
                        for fut in as_completed(futures):
                            fut.result()

                with ThreadPoolExecutor(max_workers=args.question_concurrency) as outer_pool:
                    futures = [outer_pool.submit(process_question, group) for group in tasks_by_question]
                    for fut in as_completed(futures):
                        fut.result()
            elif args.concurrency <= 1:
                for qi, ci, prompt in tasks:
                    run_task(qi, ci, prompt, phase, repetition)
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [pool.submit(run_task, qi, ci, prompt, phase, repetition) for qi, ci, prompt in tasks]
                    for fut in as_completed(futures):
                        fut.result()
            round_wall_s = time.perf_counter() - round_start
            if phase == "measured":
                round_stats.append({"repetition": repetition, "num_requests": len(tasks),
                                     "wall_s": round_wall_s, "throughput_rps": len(tasks) / round_wall_s})
            print(f"round {iteration + 1}/{args.warmup + args.repetitions} ({phase}) "
                  f"done in {round_wall_s:.2f}s ({len(tasks) / round_wall_s:.1f} req/s)")

    correct = 0
    total_graded = 0
    for qi in range(len(questions)):
        lps = yes_logprobs[qi]
        if all(lp is not None for lp in lps):
            predicted = max(range(len(lps)), key=lambda i: lps[i])
            correct += int(predicted == questions[qi]["target_index"])
            total_graded += 1

    measured = [s for s in samples if s["phase"] == "measured"]
    successes = [s["wall_ms"] for s in measured if s["success"]]
    errors = [s for s in measured if not s["success"]]
    by_candidates = {}
    for s in measured:
        if s["success"]:
            by_candidates.setdefault(s["num_candidates"], []).append(s["wall_ms"])

    # Question-level "time to decision" under concurrency: a question's candidates
    # are dispatched independently (interleaved with other questions' candidates
    # in the thread pool), so per-question latency is NOT the sum of its candidate
    # wall_ms values (that double-counts parallel work). Instead, using the
    # start_ts/end_ts wall-clock timestamps recorded per request, we take the span
    # from the first candidate's start to the last candidate's end for each
    # (question, repetition) -- i.e. "how long a caller waits if it fires all N
    # candidate requests concurrently and waits for the slowest one to return."
    # Only computed for repetitions where every candidate succeeded.
    by_question_rep = collections.defaultdict(list)
    for s in measured:
        by_question_rep[(s["question_id"], s["repetition"])].append(s)

    question_latencies_ms = []
    question_latencies_by_k = collections.defaultdict(list)
    for (qid, rep), rows in by_question_rep.items():
        if not all(r["success"] for r in rows):
            continue
        span_ms = (max(r["end_ts"] for r in rows) - min(r["start_ts"] for r in rows)) * 1000
        question_latencies_ms.append(span_ms)
        question_latencies_by_k[rows[0]["num_candidates"]].append(span_ms)

    summary = {
        "server": args.server,
        "concurrency": args.concurrency,
        "question_concurrency": args.question_concurrency,
        "throughput_rps": {
            "mean": statistics.mean(r["throughput_rps"] for r in round_stats) if round_stats else None,
            "per_round": round_stats,
        },
        "question_throughput_rps": {
            # Questions fully resolved per second = candidate throughput / avg
            # candidates per question. A "question" completes only once all of
            # its candidates have returned, so this is the more meaningful rate
            # for question-level comparisons (see latency_ms_per_question below).
            "mean": (
                (statistics.mean(r["throughput_rps"] for r in round_stats) / (len(tasks) / len(questions)))
                if round_stats else None
            ),
        },
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions": len(questions),
        "num_measured_requests": len(measured),
        "num_errors": len(errors),
        "latency_ms": {
            "p50": percentile(successes, .5),
            "p95": percentile(successes, .95),
            "p99": percentile(successes, .99),
            "min": min(successes) if successes else None,
            "max": max(successes) if successes else None,
            "mean": statistics.mean(successes) if successes else None,
            "stdev": statistics.pstdev(successes) if len(successes) > 1 else None,
        },
        "latency_ms_by_candidate_count": {
            str(k): {
                "count": len(v), "p50": percentile(v, .5), "p95": percentile(v, .95),
                "mean": statistics.mean(v),
            } for k, v in sorted(by_candidates.items())
        },
        "latency_ms_per_question": {
            "description": "Span from first candidate's start to last candidate's end, "
                            "per (question, repetition); only counted where all candidates "
                            "succeeded. Meaningful at any concurrency (unlike summing "
                            "candidate wall_ms, which only equals wall-clock time at "
                            "concurrency=1).",
            "count": len(question_latencies_ms),
            "p50": percentile(question_latencies_ms, .5),
            "p95": percentile(question_latencies_ms, .95),
            "p99": percentile(question_latencies_ms, .99),
            "min": min(question_latencies_ms) if question_latencies_ms else None,
            "max": max(question_latencies_ms) if question_latencies_ms else None,
            "mean": statistics.mean(question_latencies_ms) if question_latencies_ms else None,
            "stdev": statistics.pstdev(question_latencies_ms) if len(question_latencies_ms) > 1 else None,
        },
        "latency_ms_per_question_by_candidate_count": {
            str(k): {
                "count": len(v), "p50": percentile(v, .5), "p95": percentile(v, .95),
                "mean": statistics.mean(v),
            } for k, v in sorted(question_latencies_by_k.items())
        },
        "accuracy_proxy": {
            "description": "argmax P(yes) over candidates vs. dataset target, on questions "
                            "where every candidate returned a valid yes logprob",
            "correct": correct, "graded": total_graded,
            "accuracy": (correct / total_graded) if total_graded else None,
        },
    }
    with (output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(samples)} samples to {output / 'samples.jsonl'}")
    print(f"Wrote summary to {output / 'summary.json'}")


if __name__ == "__main__":
    main()
