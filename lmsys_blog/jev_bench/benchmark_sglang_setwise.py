"""Benchmark sglang's SETWISE scoring on the Open-Jev dataset (patched sglang only).

This is an "improved SIS": instead of SIS's N independent `items` in one
/v1/score call (each candidate tokenized/prefilled as its own sequence), this
packs ALL of a question's candidates into a SINGLE item, with one dedicated
"extraction token" anchor placed right after each candidate's own Yes/No
prompt. The server pools the LM head's label-token (Yes/No) logprobs AT each
anchor position in one prefill, so `scores` comes back as an
[1 item][N anchors x 2 labels] nested list -- one row per candidate, in the
order its anchor appears in the item.

Anchor placement is INTERLEAVED (right after each candidate's own text), not
grouped at the end -- this keeps each candidate's own Yes/No decision
immediately following only its own proposed-answer text (plus the shared
query prefix and any earlier candidates, since attention is causal), mirroring
SIS/MIS's per-candidate phrasing as closely as possible while still packing
everything into one fused item.

Requires a sglang build with setwise scoring support (score_extraction_token
on /v1/score) -- NOT available in stock sglang 0.5.20; this project applied
https://github.com/sgl-project/sglang/pull/38965 (SequenceClassification
setwise) + https://github.com/sundar24295s/sglang/pull/2 (CausalLM
extension) on top -- see this repo's README for how to obtain and build it.
See SUMMARY_SETWISE.md for the patch/install notes.

Server must be launched WITHOUT --enable-mis (this project's fused/MIS
setwise path crashes with a CUDA illegal-memory-access in our environment --
see SUMMARY_SETWISE.md). Batched setwise only requires:
    python -m sglang.launch_server --model-path <model> \
        --attention-backend flashinfer --disable-radix-cache \
        --chunked-prefill-size -1

Setwise's --disable-radix-cache requirement makes it cache-invariant by
design (same reasoning as MIS) -- there is no meaningful "cold cache" state
to flush, so unlike SIS this script has no --flush-cache-each-round flag.

Usage:
    python benchmark_sglang_setwise.py \
        --dataset-dir <dataset_path> \
        --model-path <qwen3-8b_model_path> \
        --split test --num-questions 200 --warmup 2 --repetitions 20 --concurrency 32 \
        --server http://localhost:30000 --output results_setwise
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


def build_setwise_request(question, anchor_token):
    """One item, all candidates packed in, anchors grouped at the END.

    Matches the upstream PR's own documented convention (its worked example:
    "Rank the candidates. C0. C1. C2. Scores:<anchor><anchor><anchor>") --
    the server resolves anchor->candidate purely by POSITIONAL ORDER (the
    i-th occurrence of the anchor token in the sequence becomes row i of the
    output), so this is not a code-level requirement, but it is the layout
    the PR's example (and its own reference reranker model) uses.

    An earlier version of this function interleaved each candidate's anchor
    immediately after its own text (mirroring SIS/MIS's per-candidate Yes/No
    phrasing) instead of grouping them at the end. That is mechanically
    valid too (the server doesn't care where in the text an anchor sits,
    only the order they appear in), but it deviates from the PR's own
    convention and may be a worse match for a model given no specific
    setwise fine-tuning -- worth comparing empirically.

    Prompt style still mirrors Fused-Choice's prompt shape (build_fused_prompt()
    in benchmark_sglang_fused_choice.py) as closely as the setwise format
    allows: the shared instruction is stated ONCE, and candidates are listed
    as short lettered lines, keeping the two approaches' prefill length a
    fair, matched comparison."""
    n = len(question["options"])
    letters = LETTERS[:n]
    query = (
        f"Context:\n{question['state']}\n\n"
        f"Question: {question['question']}\n\n"
        f"For each candidate answer below, decide whether it is correct. "
        f"Respond with Yes or No at each marker.\n\n"
    )
    candidate_lines = "\n".join(f"{letters[i]}) {option}" for i, option in enumerate(question["options"]))
    anchors = "".join(anchor_token for _ in range(n))
    item = f"{candidate_lines}\nScores:{anchors}\n"
    return query, item


def call_setwise_api(server, query, item, label_token_ids, score_extraction_token, timeout=60):
    payload = json.dumps({
        "model": "default",
        "query": query,
        "items": [item],
        "label_token_ids": label_token_ids,
        "score_extraction_token": score_extraction_token,
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
        # Setwise nests per item: scores == [num_items][N_i x num_labels].
        # We always send exactly one item, so unwrap to the [N x num_labels]
        # per-candidate matrix directly.
        return {
            "success": True, "wall_ms": wall_ms, "scores": body["scores"][0],
            "prompt_tokens": body["usage"]["prompt_tokens"],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        return {"success": False, "wall_ms": (time.perf_counter() - begin) * 1000,
                "error_type": type(error).__name__, "error": str(error)}
    except (KeyError, IndexError) as error:
        # Malformed/unexpected response shape (e.g. patch behaves differently
        # than expected) -- surface as a failure rather than crashing the sweep.
        return {"success": False, "wall_ms": (time.perf_counter() - begin) * 1000,
                "error_type": type(error).__name__, "error": f"unexpected response shape: {error}"}


def get_label_and_anchor_ids(model_path, score_extraction_token):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    anchor_ids = tok.encode(score_extraction_token, add_special_tokens=False)
    if len(anchor_ids) != 1:
        raise ValueError(
            f"score_extraction_token {score_extraction_token!r} did not tokenize to a single "
            f"dedicated token (got {anchor_ids}); pick a token that resolves to exactly one id."
        )
    return yes_id, no_id, anchor_ids[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", required=True, help="Local model path, to derive label/anchor token ids")
    parser.add_argument("--split", default="test", choices=["train", "calibration", "validation", "test", "ood"])
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--server", default="http://localhost:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=1,
                         help="1 = sequential, one setwise /v1/score request (whole question) at a "
                              "time. >1 fires all questions' requests for a round through a thread pool.")
    parser.add_argument("--score-extraction-token", default="<|object_ref_start|>",
                         help="Dedicated single-token anchor pooled at each candidate (default: "
                              "Qwen's <|object_ref_start|>, a vision-related special token repurposed "
                              "here since it's otherwise unused in a text-only workload).")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    yes_id, no_id, anchor_id = get_label_and_anchor_ids(args.model_path, args.score_extraction_token)
    print(f"label_token_ids: Yes={yes_id} No={no_id}  score_extraction_token={args.score_extraction_token!r} "
          f"(id={anchor_id})")

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    print(f"Loaded {len(questions)} questions "
          f"({sum(len(q['options']) for q in questions)} total candidates), concurrency={args.concurrency}")
    # One task per QUESTION -- setwise scores every candidate for a question
    # in a single request, same as SIS/MIS, but via one fused item instead of
    # N separate items.
    tasks = [(qi, *build_setwise_request(q, args.score_extraction_token)) for qi, q in enumerate(questions)]

    predicted_scores = {qi: None for qi in range(len(questions))}
    samples = []
    write_lock = threading.Lock()
    round_stats = []

    with (output / "samples.jsonl").open("w") as raw_file:
        def run_task(qi, query, item, phase, repetition):
            result = call_setwise_api(args.server, query, item, [yes_id, no_id], args.score_extraction_token)
            sample = {
                "question_id": questions[qi]["id"], "source": questions[qi]["source"],
                "num_candidates": len(questions[qi]["options"]), "phase": phase, "repetition": repetition,
                **result,
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
            round_start = time.perf_counter()
            if args.concurrency <= 1:
                for qi, query, item in tasks:
                    run_task(qi, query, item, phase, repetition)
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
        if scores is not None and len(scores) == len(question["options"]):
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
        "api": "score (setwise, batched -- no MIS)",
        "concurrency": args.concurrency,
        "label_token_ids": {"yes": yes_id, "no": no_id},
        "score_extraction_token": args.score_extraction_token,
        "score_extraction_token_id": anchor_id,
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
            "description": "argmax P(yes) over candidates (from single setwise /v1/score call) vs. "
                            "dataset target, only graded when the number of returned anchor rows "
                            "matched the number of candidates sent",
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
