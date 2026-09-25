"""Open-loop (push-mode, Poisson-arrival) benchmark: Setwise Score API
(patched sglang only).

Companion to benchmark_sglang_setwise.py, but replacing that script's
CLOSED-LOOP concurrency sweep with a genuine OPEN-LOOP / push-mode load
generator: requests arrive independently according to a Poisson process at a
caller-specified target QPS. See openloop_common.py's module docstring for
the full methodology and rationale.

Setwise packs ALL of a question's candidates into a SINGLE /v1/score item,
with one dedicated "extraction token" anchor per candidate (anchors grouped
at the end of the item, matching the upstream PR's own documented
convention) -- unchanged from benchmark_sglang_setwise.py. Requires a sglang
build with setwise scoring support (NOT available in stock sglang;
see this repo's README for how to obtain and build it). Server
must be launched WITHOUT --enable-mis, with --attention-backend flashinfer
--disable-radix-cache --chunked-prefill-size -1 (see
benchmark_sglang_setwise.py's module docstring for the full rationale).

Setwise's --disable-radix-cache requirement makes it cache-invariant by
design (same reasoning as MIS) -- there is no meaningful "cold cache" state
to flush, so unlike SIS/Fused-Choice this script has no --flush-cache-interval
flag or background flush thread.

Usage:
    python benchmark_sglang_setwise_openloop.py \
        --dataset-dir <dataset_path> \
        --model-path <qwen3-8b_model_path> \
        --qps 40 --duration 60 --server http://localhost:30000 \
        --output results_openloop_setwise_qps40
"""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from openloop_common import OpenLoopRunner, build_poisson_schedule, load_questions, summarize

LETTERS = "ABCDEFGHIJKLMNOP"  # supports up to 16 options, this dataset's max


def build_setwise_request(question, anchor_token):
    """Identical to benchmark_sglang_setwise.py's build_setwise_request()."""
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
    """Identical to benchmark_sglang_setwise.py's call_setwise_api() (latency
    timing removed here -- OpenLoopRunner measures dispatch/completion itself)."""
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        # Setwise nests per item: scores == [num_items][N_i x num_labels]. We
        # always send exactly one item, so unwrap to the per-candidate matrix.
        return {"success": True, "scores": body["scores"][0], "prompt_tokens": body["usage"]["prompt_tokens"]}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as error:
        return {"success": False, "error_type": type(error).__name__, "error": str(error)}
    except (KeyError, IndexError) as error:
        return {"success": False, "error_type": type(error).__name__,
                "error": f"unexpected response shape: {error}"}


def get_label_and_anchor_ids(model_path, score_extraction_token):
    """Identical to benchmark_sglang_setwise.py's get_label_and_anchor_ids()."""
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
    parser.add_argument("--score-extraction-token", default="<|object_ref_start|>",
                         help="Dedicated single-token anchor pooled at each candidate (default: "
                              "Qwen's <|object_ref_start|>, repurposed here since it's otherwise "
                              "unused in a text-only workload).")
    parser.add_argument("--qps-per-worker", type=int, default=10,
                         help="Worker thread pool size = max(1, round(qps / this)). Default 10 matches "
                              "the project convention of qps/10 threads; use a smaller value (e.g. 2) "
                              "for heavier models whose own per-request latency approaches or exceeds "
                              "~100ms, so the worker pool itself doesn't become the dispatch "
                              "bottleneck -- see openloop_common.py's module docstring.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    yes_id, no_id, anchor_id = get_label_and_anchor_ids(args.model_path, args.score_extraction_token)
    print(f"label_token_ids: Yes={yes_id} No={no_id}  score_extraction_token={args.score_extraction_token!r} "
          f"(id={anchor_id})")

    print(f"Loading {args.num_questions} '{args.split}' questions from {args.dataset_dir} ...")
    questions = load_questions(args.dataset_dir, args.split, args.num_questions, args.seed)
    requests_by_question = [build_setwise_request(q, args.score_extraction_token) for q in questions]

    schedule = build_poisson_schedule(args.qps, args.duration, len(questions), args.seed)
    print(f"Built Poisson schedule: {len(schedule)} requests over ~{args.duration}s at target qps={args.qps}")

    def process_fn(question_index):
        query, item = requests_by_question[question_index]
        result = call_setwise_api(args.server, query, item, [yes_id, no_id], args.score_extraction_token)
        return {
            "success": result["success"],
            "num_candidates": len(questions[question_index]["options"]),
            "question_id": questions[question_index]["id"],
            "target_index": questions[question_index]["target_index"],
            "scores": result.get("scores"),
            "error": result.get("error"),
        }

    runner = OpenLoopRunner(args.qps, schedule, process_fn, qps_per_worker=args.qps_per_worker)
    records, wall_s = runner.run()

    with (output / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Accuracy proxy: argmax P(yes) over candidates, only graded when the
    # number of returned anchor rows matched the number of candidates sent.
    correct = 0
    graded = 0
    for r in records:
        scores = r.get("scores")
        if r.get("success") and scores and len(scores) == r["num_candidates"]:
            predicted = max(range(len(scores)), key=lambda i: scores[i][0])  # argmax P(yes)
            correct += int(predicted == r["target_index"])
            graded += 1

    summary = summarize(records, wall_s, args.qps, extra={
        "approach": "setwise-openloop",
        "server": args.server,
        "model_path": args.model_path,
        "label_token_ids": {"yes": yes_id, "no": no_id},
        "score_extraction_token": args.score_extraction_token,
        "score_extraction_token_id": anchor_id,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "num_questions_pool": len(questions),
        "qps_per_worker": args.qps_per_worker,
        "num_workers": runner.num_workers,
        "accuracy_proxy": {
            "description": "argmax P(yes) over candidates (from single setwise /v1/score call) vs. "
                            "dataset target, only graded when the number of returned anchor rows "
                            "matched the number of candidates sent",
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
