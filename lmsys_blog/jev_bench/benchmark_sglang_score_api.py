"""Benchmark sglang's native /v1/score API (Multi-Item Scoring) on the Open-Jev dataset.

Unlike benchmark_sglang_jev.py (one chat-completion + max_tokens=1 request per
CANDIDATE), this drives sglang's /v1/score endpoint, which fuses a shared
`query` with all of a question's candidate `items` into a single
"query<delim>item1<delim>item2<delim>..." sequence and returns label-token
(Yes/No) probabilities for every item in ONE forward pass / ONE HTTP request.
This is the intended production pattern -- see the real llm-inference-ads-
ranking-dark-3 inference-container.src, which sets:
    app.ENABLED_APIS = [Scores]
    app.LABEL_TOKEN_IDS = [<yes_id>, <no_id>]
    sglang-engine-init.{DISABLE_RADIX_CACHE,DISABLE_CUDA_GRAPH,
                         CHUNKED_PREFILL_SIZE=-1} -- all auto-implied by MIS
The server must be launched with `--attention-backend flashinfer --enable-mis`
(sglang auto-disables CUDA graph / radix cache / chunked prefill for MIS).

Usage:
    python benchmark_sglang_score_api.py \
        --dataset-dir <dataset_path> \
        --model-path <qwen3-8b_model_path> \
        --split test --num-questions 200 --warmup 2 --repetitions 20 --concurrency 32 \
        --server http://localhost:30000 --output results_score_api
"""
import argparse
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
    sample = df.sample(n=min(num_questions, len(df)), random_state=seed).to_dict(orient="records")
    questions = []
    for row in sample:
        state = json.loads(row["state_json"])
        options = list(row["options"])
        target = list(row["target"])
        questions.append({
            "id": row["id"], "source": row["source"], "state": state,
            "question": row["question"], "options": options,
            "target_index": target.index(max(target)),
        })
    return questions


def build_score_request(question):
    """query/items mirror candidate_prompts() from Open-Jev's jev/api.py, but
    split so the shared context is the `query` and only the per-candidate
    tail is a separate `item` -- letting the server fuse them server-side."""
    query = f"Context:\n{question['state']}\n\nQuestion: {question['question']}\n"
    items = [
        f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in question["options"]
    ]
    return query, items


def call_score_api(server, query, items, label_token_ids, timeout=60):
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
    begin = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        wall_ms = (time.perf_counter() - begin) * 1000
        return {
            "success": True, "wall_ms": wall_ms, "scores": body["scores"],
            "prompt_tokens": body["usage"]["prompt_tokens"],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        return {"success": False, "wall_ms": (time.perf_counter() - begin) * 1000,
                "error_type": type(error).__name__, "error": str(error)}


def flush_cache(server, timeout=30):
    """POST /flush_cache -- clears sglang's RadixAttention prefix-cache tree.

    Only meaningful for SIS (Score API without --enable-mis); MIS always runs
    with radix cache force-disabled, so flushing has no effect there. Used to
    force every round to be a genuinely cold-cache measurement, removing the
    cross-round prefix-reuse benefit from replaying the identical question
    set every round (see --flush-cache-each-round).
    """
    req = urllib.request.Request(f"{server}/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def get_label_token_ids(model_path):
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
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--server", default="http://localhost:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=1,
                         help="1 = sequential, one /v1/score request (whole question) at a time. "
                              ">1 fires all questions' score requests for a round through a thread pool.")
    parser.add_argument("--flush-cache-each-round", action="store_true",
                         help="POST /flush_cache before every round (including warmup rounds), so "
                              "every round is a genuinely cold radix-cache measurement. No-op under "
                              "MIS (--enable-mis always force-disables radix cache server-side).")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    yes_id, no_id = get_label_token_ids(args.model_path)
    print(f"label_token_ids: Yes={yes_id} No={no_id}")

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    print(f"Loaded {len(questions)} questions "
          f"({sum(len(q['options']) for q in questions)} total candidates), concurrency={args.concurrency}")
    # One task per QUESTION now (not per candidate) -- the score API scores
    # every candidate for a question in a single request.
    tasks = [(qi, *build_score_request(q)) for qi, q in enumerate(questions)]

    predicted_scores = {qi: None for qi in range(len(questions))}
    samples = []
    write_lock = threading.Lock()
    round_stats = []

    with (output / "samples.jsonl").open("w") as raw_file:
        def run_task(qi, query, items, phase, repetition):
            result = call_score_api(args.server, query, items, [yes_id, no_id])
            sample = {
                "question_id": questions[qi]["id"], "source": questions[qi]["source"],
                "num_candidates": len(items), "phase": phase, "repetition": repetition, **result,
            }
            with write_lock:
                samples.append(sample)
                raw_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                raw_file.flush()
            if phase == "measured" and result["success"]:
                predicted_scores[qi] = result["scores"]
            return sample

        for iteration in range(args.warmup + args.repetitions):
            phase = "warmup" if iteration < args.warmup else "measured"
            repetition = iteration if phase == "warmup" else iteration - args.warmup
            if args.flush_cache_each_round:
                flush_cache(args.server)
                time.sleep(1)
            round_start = time.perf_counter()
            if args.concurrency <= 1:
                for qi, query, items in tasks:
                    run_task(qi, query, items, phase, repetition)
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [pool.submit(run_task, qi, q, it, phase, repetition) for qi, q, it in tasks]
                    for fut in as_completed(futures):
                        fut.result()
            round_wall_s = time.perf_counter() - round_start
            if phase == "measured":
                round_stats.append({"repetition": repetition, "num_requests": len(tasks),
                                     "wall_s": round_wall_s, "throughput_rps": len(tasks) / round_wall_s})
            print(f"round {iteration + 1}/{args.warmup + args.repetitions} ({phase}) "
                  f"done in {round_wall_s:.2f}s ({len(tasks) / round_wall_s:.1f} req/s, "
                  f"1 req == 1 full question w/ all candidates)")

    correct = 0
    total_graded = 0
    for qi, question in enumerate(questions):
        scores = predicted_scores[qi]
        if scores is not None:
            predicted = max(range(len(scores)), key=lambda i: scores[i][0])  # argmax P(yes)
            correct += int(predicted == question["target_index"])
            total_graded += 1

    measured = [s for s in samples if s["phase"] == "measured"]
    successes = [s["wall_ms"] for s in measured if s["success"]]
    errors = [s for s in measured if not s["success"]]
    by_candidates = {}
    for s in measured:
        if s["success"]:
            by_candidates.setdefault(s["num_candidates"], []).append(s["wall_ms"])

    summary = {
        "server": args.server,
        "api": "score (MIS)",
        "concurrency": args.concurrency,
        "label_token_ids": {"yes": yes_id, "no": no_id},
        "throughput_rps": {
            "mean": statistics.mean(r["throughput_rps"] for r in round_stats) if round_stats else None,
            "per_round": round_stats,
        },
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions": len(questions),
        "num_measured_requests": len(measured),
        "num_errors": len(errors),
        "latency_ms_per_question": {
            "p50": percentile(successes, .5), "p95": percentile(successes, .95),
            "p99": percentile(successes, .99),
            "min": min(successes) if successes else None, "max": max(successes) if successes else None,
            "mean": statistics.mean(successes) if successes else None,
            "stdev": statistics.pstdev(successes) if len(successes) > 1 else None,
        },
        "latency_ms_by_candidate_count": {
            str(k): {"count": len(v), "p50": percentile(v, .5), "p95": percentile(v, .95), "mean": statistics.mean(v)}
            for k, v in sorted(by_candidates.items())
        },
        "accuracy_proxy": {
            "description": "argmax P(yes) over candidates (from single /v1/score call) vs. dataset target",
            "correct": correct, "graded": total_graded,
            "accuracy": (correct / total_graded) if total_graded else None,
        },
    }
    with (output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(samples)} samples to {output / 'samples.jsonl'}")


if __name__ == "__main__":
    main()
