"""Benchmark Marlin-2B video captioning through an SGLang OpenAI server."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


DEFAULT_PROMPT = (
    "Provide a spatial description of this clip followed by time-ranged events.\n"
    "For each event, give the time range as <start - end> and a short description."
)


def load_video_urls(args):
    urls = []
    if args.video_url:
        urls.extend(args.video_url)
    if args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    if not urls:
        raise ValueError("Pass at least one --video-url or --video-list entry.")
    return urls


def request_once(args, video_url, run_index, warmup):
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    start = time.perf_counter()
    try:
        response = requests.post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=args.timeout,
        )
        elapsed_s = time.perf_counter() - start
        result = response.json()
    except Exception as exc:
        elapsed_s = time.perf_counter() - start
        return {
            "ok": False,
            "warmup": warmup,
            "run_index": run_index,
            "video_url": video_url,
            "elapsed_s": elapsed_s,
            "error": f"{type(exc).__name__}: {exc}",
        }

    usage = result.get("usage") or {}
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    row = {
        "ok": response.ok,
        "status_code": response.status_code,
        "warmup": warmup,
        "run_index": run_index,
        "video_url": video_url,
        "elapsed_s": elapsed_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_s": (
            completion_tokens / elapsed_s if completion_tokens and elapsed_s else None
        ),
        "total_tokens_per_s": total_tokens / elapsed_s if total_tokens and elapsed_s else None,
        "text": message.get("content"),
    }
    if not response.ok:
        row["error"] = result
    return row


def summarize(rows):
    measured = [row for row in rows if row.get("ok") and not row.get("warmup")]
    if not measured:
        return {"num_measured": 0}

    def metric(name):
        vals = [row[name] for row in measured if row.get(name) is not None]
        if not vals:
            return None
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "num_measured": len(measured),
        "elapsed_s": metric("elapsed_s"),
        "completion_tokens_per_s": metric("completion_tokens_per_s"),
        "total_tokens_per_s": metric("total_tokens_per_s"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument(
        "--video-url",
        action="append",
        default=[],
        help="Video URL to caption. Can be passed more than once.",
    )
    parser.add_argument("--video-list", help="Text file with one video URL per line.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--output",
        default="benchmark/marlin_video/sglang_marlin_results.jsonl",
        help="JSONL output path.",
    )
    args = parser.parse_args()

    videos = load_video_urls(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with output_path.open("a", encoding="utf-8") as f:
        for video_index, video_url in enumerate(videos):
            for run_index in range(args.warmup + args.runs):
                warmup = run_index < args.warmup
                row = request_once(args, video_url, run_index, warmup)
                row["created_at"] = datetime.now(timezone.utc).isoformat()
                row["max_tokens"] = args.max_tokens
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                f.flush()

                label = "warmup" if warmup else "run"
                if row.get("ok"):
                    completion_tps = row.get("completion_tokens_per_s")
                    completion_tps_str = (
                        f"{completion_tps:.2f}" if completion_tps is not None else "n/a"
                    )
                    print(
                        f"{label} video={video_index} "
                        f"idx={run_index} elapsed={row['elapsed_s']:.3f}s "
                        f"completion_tps={completion_tps_str}"
                    )
                else:
                    print(
                        f"{label} video={video_index} "
                        f"idx={run_index} failed elapsed={row['elapsed_s']:.3f}s "
                        f"error={row.get('error')}"
                    )

    print(json.dumps({"summary": summarize(rows), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
