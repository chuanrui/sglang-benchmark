"""Benchmark a FUSED multiple-choice Generate API request on the Open-Jev dataset.

A fourth approach alongside Generate API (benchmark_sglang_jev.py, N requests
per question), SIS, and MIS (benchmark_sglang_score_api.py, 1 fused /v1/score
request per question): this fuses all of a question's candidates into a
SINGLE /v1/chat/completions request, with each option inlined and labeled by
a letter (A, B, C, ...). The model generates exactly one output token
(max_tokens=1), and rather than trusting the sampled token verbatim, we read
the logprobs of every candidate LETTER at that output position from
`top_logprobs` and take the argmax -- the same "don't trust the sample, trust
the logprobs" pattern the other three approaches already use for Yes/No.

This needs no /v1/score endpoint and no --enable-mis; it is a single ordinary
chat-completions call per question, so in principle it should be usable
against any Generate-API-only deployment. Its two open questions vs. SIS/MIS
are (a) whether the model reliably ranks the correct letter highest even
though it isn't explicitly trained for this exact prompt format, and (b)
whether a single shared-prefix forward pass here is actually as fast as
SIS/MIS's fused single request (this script measures that).

No tokenizer / model-path needed: unlike the Score API (which requires exact
label_token_ids in the request payload), the Generate API returns each
top_logprobs entry as decoded text, so we match candidate letters by string
comparison, not token id -- consistent with how benchmark_sglang_jev.py
already matches "yes"/"no" by text.

Usage:
    python benchmark_sglang_fused_choice.py \
        --dataset-dir <dataset_path> \
        --split test --num-questions 200 --warmup 2 --repetitions 20 --concurrency 32 \
        --server http://localhost:30000 --output results_fused_choice
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

LETTERS = "ABCDEFGHIJKLMNOP"  # supports up to 16 options, this dataset's max


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
    if df.empty:
        raise ValueError(f"No rows with kind={kind!r} in split={split!r}")
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


def build_fused_prompt(question):
    """One prompt covering all of a question's candidates via lettered options.

    Polished vs. the initial sanity-check version: options are introduced
    under an explicit "Options:" heading (clearer visual separation from the
    context/question than inline running text), and the closing instruction
    is tightened to reduce free-form completions (e.g. "To" instead of a
    letter) by explicitly stating the response must be exactly one letter,
    not just "answer with the letter".
    """
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
    """Match candidate letters against top_logprobs entries by decoded text
    (case-insensitive, whitespace-stripped) -- mirrors how the plain Generate
    API script (benchmark_sglang_jev.py) matches "yes"/"no" by text rather
    than token id, since /v1/chat/completions returns decoded strings, not
    token ids, in its logprobs payload."""
    result = {letter: None for letter in letters}
    for entry in top_logprobs:
        text = entry.get("token", "").strip().upper()
        if text in result and result[text] is None:
            result[text] = entry.get("logprob")
    return result


def call_sglang_fused(server, prompt, letters, top_logprobs_n, timeout=60):
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs_n,
        # Qwen3 defaults to emitting a <think> block first; disable so the
        # single generated token is the letter choice itself.
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
        wall_ms = (time.perf_counter() - begin) * 1000
        choice = body["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("content", [{}])
        top = token_logprobs[0].get("top_logprobs", []) if token_logprobs else []
        letter_logprobs = extract_letter_logprobs(top, letters)
        return {
            "success": True,
            "wall_ms": wall_ms,
            "sampled_token": choice["message"]["content"],
            "letter_logprobs": letter_logprobs,
            "prompt_tokens": body["usage"]["prompt_tokens"],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        return {"success": False, "wall_ms": (time.perf_counter() - begin) * 1000,
                "error_type": type(error).__name__, "error": str(error)}


def flush_cache(server, timeout=30):
    """POST /flush_cache -- clears sglang's RadixAttention prefix-cache tree.
    See benchmark_sglang_score_api.py's flush_cache for the full rationale;
    identical mechanism, this approach has no MIS-equivalent mode so this
    flag is always meaningful here (unlike for MIS, where it's a no-op)."""
    req = urllib.request.Request(f"{server}/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "calibration", "validation", "test", "ood"])
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--server", default="http://localhost:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=1,
                         help="1 = sequential, one fused-choice request (whole question) at a time. "
                              ">1 fires all questions' requests for a round through a thread pool.")
    parser.add_argument("--top-logprobs", type=int, default=20,
                         help="Must be >= the max candidate count in the sample (this dataset maxes "
                              "at 16); set with margin so the correct letter is very likely captured "
                              "even when the model doesn't rank it in the top few tokens.")
    parser.add_argument("--flush-cache-each-round", action="store_true",
                         help="POST /flush_cache before every round (including warmup rounds), so "
                              "every round is a genuinely cold radix-cache measurement.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    print(f"Loaded {len(questions)} questions "
          f"(candidate counts: {sorted(set(len(q['options']) for q in questions))}), "
          f"concurrency={args.concurrency}")

    # One task per QUESTION -- a single fused request covers all candidates.
    tasks = [(qi, build_fused_prompt(q), LETTERS[:len(q["options"])]) for qi, q in enumerate(questions)]

    predicted_letter_logprobs = {qi: None for qi in range(len(questions))}
    top1_letter_hits = 0  # diagnostic: how often the greedily-SAMPLED token was itself a valid letter
    samples = []
    write_lock = threading.Lock()
    round_stats = []

    with (output / "samples.jsonl").open("w") as raw_file:
        def run_task(qi, prompt, letters, phase, repetition):
            nonlocal top1_letter_hits
            result = call_sglang_fused(args.server, prompt, letters, args.top_logprobs)
            sample = {
                "question_id": questions[qi]["id"], "source": questions[qi]["source"],
                "num_candidates": len(letters), "phase": phase, "repetition": repetition, **result,
            }
            with write_lock:
                samples.append(sample)
                raw_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                raw_file.flush()
                if phase == "measured" and result["success"] and result["sampled_token"].strip().upper() in letters:
                    top1_letter_hits += 1
            if phase == "measured" and result["success"]:
                predicted_letter_logprobs[qi] = result["letter_logprobs"]
            return sample

        for iteration in range(args.warmup + args.repetitions):
            phase = "warmup" if iteration < args.warmup else "measured"
            repetition = iteration if phase == "warmup" else iteration - args.warmup
            if args.flush_cache_each_round:
                flush_cache(args.server)
                time.sleep(1)
            round_start = time.perf_counter()
            if args.concurrency <= 1:
                for qi, prompt, letters in tasks:
                    run_task(qi, prompt, letters, phase, repetition)
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [pool.submit(run_task, qi, p, l, phase, repetition) for qi, p, l in tasks]
                    for fut in as_completed(futures):
                        fut.result()
            round_wall_s = time.perf_counter() - round_start
            if phase == "measured":
                round_stats.append({"repetition": repetition, "num_requests": len(tasks),
                                     "wall_s": round_wall_s, "throughput_rps": len(tasks) / round_wall_s})
            print(f"round {iteration + 1}/{args.warmup + args.repetitions} ({phase}) "
                  f"done in {round_wall_s:.2f}s ({len(tasks) / round_wall_s:.1f} req/s, "
                  f"1 req == 1 full question w/ all candidates)")

    # Accuracy proxy: argmax over candidate-letter logprobs (NOT the sampled
    # token) vs. dataset target, only on questions where every candidate
    # letter's logprob was captured within top_logprobs (mirrors the other
    # three approaches' "all-or-nothing" grading rule).
    correct = 0
    total_graded = 0
    total_ungraded_missing_logprob = 0
    for qi, question in enumerate(questions):
        lp = predicted_letter_logprobs[qi]
        if lp is None:
            continue
        if any(v is None for v in lp.values()):
            total_ungraded_missing_logprob += 1
            continue
        letters_sorted = sorted(lp.keys())
        predicted_letter = max(letters_sorted, key=lambda letter: lp[letter])
        predicted_index = LETTERS.index(predicted_letter)
        correct += int(predicted_index == question["target_index"])
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
        "api": "fused-choice (Generate API, single fused MCQ request per question)",
        "concurrency": args.concurrency,
        "top_logprobs": args.top_logprobs,
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
            "description": "One fused request already covers the whole question, so this is "
                            "directly the per-request latency (no per-question aggregation needed, "
                            "unlike Generate API's N-requests-per-question case).",
            "p50": percentile(successes, .5), "p95": percentile(successes, .95),
            "p99": percentile(successes, .99),
            "min": min(successes) if successes else None, "max": max(successes) if successes else None,
            "mean": statistics.mean(successes) if successes else None,
            "stdev": statistics.pstdev(successes) if len(successes) > 1 else None,
        },
        "latency_ms_by_candidate_count": {
            str(k): {
                "count": len(v), "p50": percentile(v, .5), "p95": percentile(v, .95),
                "mean": statistics.mean(v),
            } for k, v in sorted(by_candidates.items())
        },
        "accuracy_proxy": {
            "description": "argmax over candidate-LETTER logprobs (not the greedily sampled token) "
                            "vs. dataset target, on questions where every candidate letter's logprob "
                            "was found within top_logprobs.",
            "correct": correct, "graded": total_graded,
            "accuracy": (correct / total_graded) if total_graded else None,
            "ungraded_missing_a_letter_logprob": total_ungraded_missing_logprob,
        },
        "diagnostic_top1_sampled_token_was_a_valid_letter": {
            "description": "How often the model's actual greedily-sampled token (what a naive "
                            "caller ignoring logprobs would use) was itself one of the valid letters "
                            "for that question -- a lower bound on how 'well-behaved' this prompt "
                            "format is, independent of the logprob-based accuracy_proxy above.",
            "count": top1_letter_hits, "out_of": len(measured),
            "rate": (top1_letter_hits / len(measured)) if measured else None,
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
