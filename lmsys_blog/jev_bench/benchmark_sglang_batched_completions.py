"""Benchmark a REAL BATCHED /v1/completions Generate request on the Open-Jev dataset.

A fifth approach alongside Generate API (benchmark_sglang_jev.py, N separate
/v1/chat/completions requests per question), SIS/MIS (benchmark_sglang_score_api.py,
1 fused /v1/score request per question), and Fused-Choice
(benchmark_sglang_fused_choice.py, 1 /v1/chat/completions request per question with
lettered options inlined): this uses the legacy OpenAI-compatible
`POST /v1/completions` endpoint's native batch support -- unlike
`/v1/chat/completions`, whose `messages` field is a single conversation with no
batch form, `/v1/completions`'s `prompt` field accepts a LIST of independent
prompt strings (confirmed in sglang's own protocol.py:
`prompt: Union[List[int], List[List[int]], str, List[str]]`). Sending all of a
question's N candidate prompts as one list produces ONE HTTP request and ONE
response containing N independently-indexed choices -- a genuine client-side
batch call, not N round trips and not a query/KV-fused single sequence like
MIS/Setwise. Each candidate is still tokenized/prefilled as its own sequence
server-side (no shared-prefix fusion), so this trades away MIS/Setwise's
prefill-amortization win in exchange for needing no /v1/score endpoint, no
--enable-mis, and no patched sglang -- ordinary stock sglang, one call per
question.

Candidate rendering is identical to benchmark_sglang_jev.py (isolated
"Proposed answer: X\nIs this proposed answer correct? Answer Yes or No."
prompts, one per option) so that a Generate-API accuracy/latency comparison
against SIS/MIS isn't confounded by a different prompt format (unlike
Fused-Choice's lettered-MCQ prompt, which is a genuinely different task
framing for the model).

IMPORTANT: unlike /v1/chat/completions, /v1/completions does NOT apply the
model's chat template automatically -- it is a raw text-completion endpoint.
Sending our plain "Context:...\nProposed answer:..." text through it produces
free-form text continuation (confirmed empirically: Qwen3-8B replied with
continuations like " Then" instead of "Yes"/"No"), not an instruction
response. This script therefore requires --model-path and applies the
tokenizer's own chat template client-side (mirroring /v1/chat/completions'
server-side templating, including enable_thinking=False) before sending each
candidate string into the batched prompt list.

Usage:
    python benchmark_sglang_batched_completions.py \
        --dataset-dir <dataset_path> \
        --model-path <qwen3-8b_model_path> \
        --split test --num-questions 200 --warmup 2 --repetitions 20 --concurrency 32 \
        --server http://localhost:30000 --output results_batched_completions
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


def candidate_prompts(question):
    """Render each candidate option as an isolated yes/no prompt -- identical
    to benchmark_sglang_jev.py's candidate_prompts(), so the only difference
    from plain Generate API is the transport (one batched request vs. N
    separate requests), not the prompt content."""
    prefix = f"Context:\n{question['state']}\n\nQuestion: {question['question']}\n"
    return [
        prefix + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in question["options"]
    ]


def get_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def apply_chat_template(tokenizer, prompt):
    """/v1/completions is a raw text-completion endpoint -- it does NOT apply
    the chat template the way /v1/chat/completions does server-side. Apply it
    client-side so the model sees the same instruct-formatted input either
    way (including disabling Qwen3's default <think> block, matching the
    chat_template_kwargs={"enable_thinking": False} used by
    benchmark_sglang_jev.py / benchmark_sglang_fused_choice.py)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def call_sglang_batched(server, prompts, top_logprobs_n, timeout=60):
    """POST /v1/completions with prompt=[p1, ..., pN] -- one HTTP request,
    N independently-indexed choices back. Each choice's logprobs.top_logprobs
    is a list of {token: logprob} dicts, one per generated position; with
    max_tokens=1 there is exactly one such dict per choice."""
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
    begin = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        end = time.perf_counter()
        wall_ms = (end - begin) * 1000
        choices = body["choices"]
        if len(choices) != len(prompts):
            raise ValueError(f"expected {len(prompts)} choices, got {len(choices)}")
        by_index = {c["index"]: c for c in choices}
        if sorted(by_index.keys()) != list(range(len(prompts))):
            raise ValueError(f"choice indices not exactly 0..{len(prompts) - 1}: {sorted(by_index.keys())}")
        yes_logprobs = []
        sampled_tokens = []
        for i in range(len(prompts)):
            choice = by_index[i]
            sampled_tokens.append(choice.get("text"))
            top = None
            lp = choice.get("logprobs")
            if lp and lp.get("top_logprobs"):
                top = lp["top_logprobs"][0]  # dict: token -> logprob, for the single generated token
            yes_lp = None
            if top:
                for tok, val in top.items():
                    if tok.strip().lower() == "yes":
                        yes_lp = val
                        break
            yes_logprobs.append(yes_lp)
        return {
            "success": True,
            "wall_ms": wall_ms,
            "sampled_tokens": sampled_tokens,
            "yes_logprobs": yes_logprobs,
            "prompt_tokens_total": body["usage"]["prompt_tokens"],
            "completion_tokens_total": body["usage"]["completion_tokens"],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError,
            ValueError, KeyError) as error:
        end = time.perf_counter()
        return {"success": False, "wall_ms": (end - begin) * 1000,
                "error_type": type(error).__name__, "error": str(error)}


def flush_cache(server, timeout=30):
    """POST /flush_cache -- clears sglang's RadixAttention prefix-cache tree.
    See benchmark_sglang_score_api.py's flush_cache for the full rationale."""
    req = urllib.request.Request(f"{server}/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True,
                         help="Local model path; used to load the tokenizer and apply its chat "
                              "template client-side (required -- /v1/completions does not apply it "
                              "server-side, unlike /v1/chat/completions).")
    parser.add_argument("--split", default="test", choices=["train", "calibration", "validation", "test", "ood"])
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--server", default="http://localhost:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Number of in-flight QUESTIONS (each is already 1 batched HTTP "
                              "request covering all of its candidates). 1 = sequential, one "
                              "question's batched request at a time. >1 fires all questions' "
                              "batched requests for a round through a thread pool of this size -- "
                              "the same 'concurrency = question granularity' convention used by "
                              "benchmark_sglang_score_api.py, benchmark_sglang_fused_choice.py, and "
                              "benchmark_sglang_setwise.py.")
    parser.add_argument("--top-logprobs", type=int, default=5,
                         help="logprobs=N on the /v1/completions request; must be large enough that "
                              "'Yes'/'yes' reliably appears in top_logprobs for a well-behaved model.")
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

    print(f"Loading tokenizer from {args.model_path} to apply chat template client-side ...")
    tokenizer = get_tokenizer(args.model_path)

    # One task per QUESTION -- a single batched /v1/completions request covers
    # all of its candidates (unlike benchmark_sglang_jev.py, which flattens to
    # one task per (question, candidate)). Each candidate string is passed
    # through the tokenizer's own chat template before being placed in the
    # batched prompt list, since /v1/completions won't do this for us.
    tasks = [
        (qi, [apply_chat_template(tokenizer, p) for p in candidate_prompts(q)])
        for qi, q in enumerate(questions)
    ]

    predicted_yes_logprobs = {qi: None for qi in range(len(questions))}
    samples = []
    write_lock = threading.Lock()
    round_stats = []

    with (output / "samples.jsonl").open("w") as raw_file:
        def run_task(qi, prompts, phase, repetition):
            result = call_sglang_batched(args.server, prompts, args.top_logprobs)
            sample = {
                "question_id": questions[qi]["id"], "source": questions[qi]["source"],
                "num_candidates": len(prompts), "phase": phase, "repetition": repetition, **result,
            }
            with write_lock:
                samples.append(sample)
                raw_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                raw_file.flush()
            if phase == "measured" and result["success"]:
                predicted_yes_logprobs[qi] = result["yes_logprobs"]
            return sample

        for iteration in range(args.warmup + args.repetitions):
            phase = "warmup" if iteration < args.warmup else "measured"
            repetition = iteration if phase == "warmup" else iteration - args.warmup
            if args.flush_cache_each_round:
                flush_cache(args.server)
                time.sleep(1)
            round_start = time.perf_counter()
            if args.concurrency <= 1:
                for qi, prompts in tasks:
                    run_task(qi, prompts, phase, repetition)
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [pool.submit(run_task, qi, prompts, phase, repetition) for qi, prompts in tasks]
                    for fut in as_completed(futures):
                        fut.result()
            round_wall_s = time.perf_counter() - round_start
            if phase == "measured":
                round_stats.append({"repetition": repetition, "num_requests": len(tasks),
                                     "wall_s": round_wall_s, "throughput_rps": len(tasks) / round_wall_s})
            print(f"round {iteration + 1}/{args.warmup + args.repetitions} ({phase}) "
                  f"done in {round_wall_s:.2f}s ({len(tasks) / round_wall_s:.1f} req/s, "
                  f"1 req == 1 batched question w/ all candidates)")

    # Accuracy proxy: argmax over candidates' P(yes) logprobs vs. dataset
    # target, only on questions where every candidate returned a yes logprob
    # (mirrors SIS/MIS/Fused-Choice/Setwise's "all-or-nothing" grading rule).
    correct = 0
    total_graded = 0
    total_ungraded_missing_logprob = 0
    for qi, question in enumerate(questions):
        lps = predicted_yes_logprobs[qi]
        if lps is None:
            continue
        if any(lp is None for lp in lps):
            total_ungraded_missing_logprob += 1
            continue
        predicted_index = max(range(len(lps)), key=lambda i: lps[i])
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
        "model_path": args.model_path,
        "api": "batched-completions (Generate API, one /v1/completions request "
               "with prompt=[N candidates] per question)",
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
            "description": "One batched request already covers the whole question (all N "
                            "candidates), so this is directly the per-request latency -- no "
                            "per-question aggregation needed, unlike plain Generate API's "
                            "N-separate-requests case (benchmark_sglang_jev.py).",
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
            "description": "argmax P(yes) over candidates (from one batched /v1/completions call) "
                            "vs. dataset target, on questions where every candidate returned a yes "
                            "logprob.",
            "correct": correct, "graded": total_graded,
            "accuracy": (correct / total_graded) if total_graded else None,
            "ungraded_missing_a_yes_logprob": total_ungraded_missing_logprob,
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
