"""Shared open-loop (push-mode, Poisson-arrival) load-generation harness.

The existing benchmark scripts in this project (benchmark_sglang_jev.py,
benchmark_sglang_batched_completions.py, benchmark_sglang_score_api.py, etc.)
are all CLOSED-LOOP / "pull-mode": we fix a concurrency N, fire N requests,
wait for the round to finish, repeat. Question-level THROUGHPUT is then
*derived* from that (num_requests / wall_time), and we interpolate across
concurrency levels to find the throughput <-> latency curve. This is a
standard and widely-used methodology, but it has a well-known blind spot
(sometimes called "coordinated omission"): because the next request is only
issued once a worker is free, a closed-loop system can never accumulate a
queue in front of an overloaded server -- if the server is momentarily slow,
concurrency-based dispatch just waits, so it never observes the pile-up of
requests a real, independent-arrival production traffic pattern would create.

This module instead builds a genuine OPEN-LOOP / "push-mode" load generator:
- The caller specifies a target QPS and a duration.
- ALL requests for the whole run are pre-generated up front, each with its
  own scheduled arrival time drawn from a Poisson process (i.e. inter-arrival
  gaps are i.i.d. Exponential(1/qps)) -- this is what real, independent
  traffic looks like, not a fixed concurrency level.
- A fixed-size pool of worker threads (qps/10, per the project's convention)
  pulls requests in scheduled order and dispatches each as close to its
  scheduled time as possible. If the pool is saturated when a request's
  scheduled time arrives, the request queues (exactly like a real server's
  request queue would) -- and that queueing delay is captured, not hidden.
- We report BOTH the pure service latency (dispatch -> completion, which is
  comparable to the old closed-loop per-request latency) AND the end-to-end
  latency (scheduled-arrival -> completion, which includes queueing delay
  and is the metric that actually reflects "how long would a real caller
  sending traffic at this QPS have waited").

Usage pattern for a benchmark script built on this module:
    questions = load_questions(...)
    schedule = build_poisson_schedule(qps, duration, len(questions), seed)
    runner = OpenLoopRunner(qps, schedule, process_fn)
    records = runner.run()
    summary = summarize(records, ...)
"""
import glob
import json
import math
import os
import queue
import statistics
import threading
import time
import urllib.error
import urllib.request

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
    """Identical to the other benchmark scripts' load_questions()."""
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


def build_poisson_schedule(qps, duration_s, num_questions, seed):
    """Pre-generate every request's (question_index, scheduled_time_s) pair.

    Inter-arrival gaps are drawn i.i.d. from Exponential(1/qps) -- the
    defining property of a Poisson arrival process with rate `qps`. The
    cumulative sum of gaps gives each request's scheduled arrival time
    (seconds since the run's logical start, t=0).

    IMPORTANT: question indices are NEVER cycled/repeated within a single
    run. Each request gets a distinct question index (0, 1, 2, ...,
    total-1), so no two requests in the same run ever send the identical
    prompt text. This is required for correctness: unlike the closed-loop
    scripts (which flush the radix-cache prefix-cache tree between discrete
    rounds), an open-loop run has no natural quiescent point to flush at --
    flush_cache() requires no in-flight requests, a guarantee continuous
    Poisson traffic can never provide. If the same question were allowed to
    recur mid-run (e.g. via `i % num_questions` cycling), it could hit a
    still-warm radix-cache entry from its earlier occurrence, silently
    lowering its measured latency and contaminating the result. Capping the
    schedule at `num_questions` (never cycling) sidesteps this categorically:
    every request's text is unique, so no cross-request cache hit is even
    possible, without needing runtime cache flushing to be correct.

    If the requested total request count (round(qps * duration_s)) exceeds
    num_questions, the schedule is CAPPED at num_questions and a warning is
    printed -- the realized run will be shorter than duration_s (roughly
    num_questions / qps seconds) but will still contain num_questions
    samples, which is often a perfectly adequate sample size for percentile
    estimates. Pass a large --num-questions (the loader clips to however many
    rows actually exist) to raise this ceiling; see load_questions().
    """
    import random
    rng = random.Random(seed)
    requested_total = max(1, round(qps * duration_s))
    total = min(requested_total, num_questions)
    if total < requested_total:
        print(f"WARNING: requested {requested_total} requests (qps={qps} x duration={duration_s}s) "
              f"exceeds the {num_questions}-question pool available; capping at {total} requests to "
              f"guarantee no question repeats within this run (a repeat could hit a still-warm "
              f"radix-cache entry, since there is no quiescent point in a continuous open-loop run "
              f"to safely flush at). Effective duration will be ~{total / qps:.1f}s instead of "
              f"{duration_s}s; pass a larger --num-questions to raise this ceiling.")
    scheduled = []
    t = 0.0
    for _ in range(total):
        gap = rng.expovariate(qps)  # Exponential(rate=qps), mean gap = 1/qps
        t += gap
        scheduled.append(t)
    schedule = [(i, scheduled[i]) for i in range(total)]  # i directly -- never cycles, never repeats
    return schedule


class OpenLoopRunner:
    """Fixed-size worker-thread pool that dispatches a pre-built Poisson
    schedule as close to each request's scheduled time as possible.

    Worker count follows this project's convention: max(1, round(qps/10)).
    Workers share a single FIFO queue.Queue seeded with the whole schedule
    (already in ascending scheduled-time order); an idle worker always picks
    up the next-in-schedule request, so the pool behaves like a standard
    multi-server queue (M/M/c-ish, c = worker count) rather than a static
    per-worker partition -- a fast worker doesn't sit idle while a slow
    worker's later-scheduled requests pile up behind it.
    """

    def __init__(self, qps, schedule, process_fn, qps_per_worker=10):
        self.qps = qps
        self.schedule = schedule
        self.process_fn = process_fn
        self.num_workers = max(1, round(qps / qps_per_worker))
        self._q = queue.Queue()
        for job in schedule:
            self._q.put(job)
        self._records = []
        self._lock = threading.Lock()

    def _worker(self, start_perf):
        while True:
            try:
                question_index, scheduled_t = self._q.get_nowait()
            except queue.Empty:
                return
            now = time.perf_counter() - start_perf
            wait = scheduled_t - now
            if wait > 0:
                time.sleep(wait)
            dispatch_t = time.perf_counter() - start_perf
            result = self.process_fn(question_index)
            completion_t = time.perf_counter() - start_perf
            record = {
                "question_index": question_index,
                "scheduled_time_s": scheduled_t,
                "dispatch_time_s": dispatch_t,
                "completion_time_s": completion_t,
                "queueing_delay_ms": (dispatch_t - scheduled_t) * 1000,
                "service_latency_ms": (completion_t - dispatch_t) * 1000,
                "end_to_end_latency_ms": (completion_t - scheduled_t) * 1000,
                **result,
            }
            with self._lock:
                self._records.append(record)

    def run(self):
        print(f"OpenLoopRunner: {len(self.schedule)} requests, target qps={self.qps}, "
              f"{self.num_workers} worker threads (max(1, round(qps/10)))")
        start_perf = time.perf_counter()
        threads = [threading.Thread(target=self._worker, args=(start_perf,), daemon=True)
                   for _ in range(self.num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_s = time.perf_counter() - start_perf
        return self._records, wall_s


def flush_cache(server, timeout=30):
    req = urllib.request.Request(f"{server}/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def periodic_flush_thread(server, interval_s, stop_event):
    """Background thread that flushes sglang's radix-cache prefix tree every
    `interval_s` seconds for the duration of the run, approximating the
    "always cold" methodology the closed-loop scripts get via
    --flush-cache-each-round -- there is no discrete "round" boundary in an
    open-loop run to hook a flush onto, so we flush on a wall-clock timer
    instead. Skip entirely (don't start this thread) for MIS, which is
    cache-invariant by design and must not be flushed (flush_cache requires
    no in-flight requests, and MIS relies on radix caching being disabled
    server-side already)."""
    while not stop_event.wait(interval_s):
        try:
            flush_cache(server)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError):
            pass  # best-effort; a busy server may reject a flush mid-flight, just retry next tick


def summarize(records, wall_s, target_qps, extra=None):
    """Common summary schema shared by all three open-loop scripts.

    Reports BOTH end_to_end_latency_ms (scheduled arrival -> completion,
    includes queueing delay -- the "true" push-mode latency the user asked
    for) and service_latency_ms (dispatch -> completion, comparable to the
    old closed-loop scripts' per-request latency), so a reader can see how
    much of the true latency is queueing vs. actual service time.
    """
    successes = [r for r in records if r.get("success")]
    errors = [r for r in records if not r.get("success")]
    e2e = [r["end_to_end_latency_ms"] for r in successes]
    svc = [r["service_latency_ms"] for r in successes]
    queueing = [r["queueing_delay_ms"] for r in successes]
    achieved_qps = len(successes) / wall_s if wall_s > 0 else None

    by_candidates_e2e = {}
    by_candidates_svc = {}
    for r in successes:
        n = r.get("num_candidates")
        if n is not None:
            by_candidates_e2e.setdefault(n, []).append(r["end_to_end_latency_ms"])
            by_candidates_svc.setdefault(n, []).append(r["service_latency_ms"])

    def pct_block(values):
        return {
            "count": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95),
            "p99": percentile(values, .99), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "stdev": statistics.pstdev(values) if len(values) > 1 else None,
        }

    summary = {
        "target_qps": target_qps,
        "wall_s": wall_s,
        "achieved_qps": achieved_qps,
        "num_requests_scheduled": len(records),
        "num_success": len(successes),
        "num_errors": len(errors),
        "end_to_end_latency_ms": pct_block(e2e),
        "service_latency_ms": pct_block(svc),
        "queueing_delay_ms": pct_block(queueing),
        "end_to_end_latency_ms_by_candidate_count": {
            str(k): pct_block(v) for k, v in sorted(by_candidates_e2e.items())
        },
        "service_latency_ms_by_candidate_count": {
            str(k): pct_block(v) for k, v in sorted(by_candidates_svc.items())
        },
    }
    if extra:
        summary.update(extra)
    return summary
