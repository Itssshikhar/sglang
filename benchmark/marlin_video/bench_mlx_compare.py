"""Compare Marlin video captioning on SGLang MLX and a custom MLX runner.

This script is intended for Apple Silicon machines.  It can launch an SGLang
server with SGLANG_USE_MLX=1, send OpenAI-compatible video requests, and run a
custom MLX hybrid implementation through either a shell command template or a
Python module:function callable.
"""

import argparse
import importlib
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any

import requests


DEFAULT_PROMPT = (
    "Provide a spatial description of this clip followed by time-ranged events.\n"
    "For each event, give the time range as <start - end> and a short description."
)

DEFAULT_VIDEO_URL = (
    "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/"
    "jobs_presenting_ipod.mp4"
)

DEFAULT_MM_PROCESS_CONFIG = (
    '{"video":{"fps":2.0,"min_frames":4,"max_frames":240,"max_pixels":200704}}'
)


def load_video_urls(args: argparse.Namespace) -> list[str]:
    urls = []
    if args.video_url:
        urls.extend(args.video_url)
    if args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as f:
            urls.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    if not urls:
        urls.append(DEFAULT_VIDEO_URL)
    return urls


def build_sglang_command(args: argparse.Namespace) -> list[str]:
    if args.sglang_server_command:
        return format_command(
            args.sglang_server_command,
            {
                "model_path": args.sglang_model_path,
                "tokenizer_path": args.sglang_tokenizer_path or args.sglang_model_path,
                "served_model_name": args.sglang_model,
                "host": args.sglang_host,
                "port": args.sglang_port,
                "mm_process_config": args.mm_process_config,
            },
        )

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.sglang_model_path,
        "--served-model-name",
        args.sglang_model,
        "--trust-remote-code",
        "--enable-multimodal",
        "--disable-cuda-graph",
        "--host",
        args.sglang_host,
        "--port",
        str(args.sglang_port),
        "--mm-process-config",
        args.mm_process_config,
    ]
    if args.sglang_tokenizer_path:
        cmd.extend(["--tokenizer-path", args.sglang_tokenizer_path])
    if args.disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if args.sglang_quantization:
        cmd.extend(["--quantization", args.sglang_quantization])
    if args.sglang_extra_arg:
        for value in args.sglang_extra_arg:
            cmd.extend(shlex.split(value))
    return cmd


def format_command(template: str, values: dict[str, Any]) -> list[str]:
    quoted_values = {key: shlex.quote(str(value)) for key, value in values.items()}
    rendered = template.format(**quoted_values)
    return shlex.split(rendered)


def wait_for_sglang(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    base = base_url.rstrip("/")
    # Newer SGLang serves model info at /get_model_info on the root, not /v1.
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    urls = [f"{base}/model_info", f"{root}/get_model_info", f"{root}/health"]
    while time.monotonic() < deadline:
        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                if response.ok:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready: {last_error}")


def build_sglang_payload(
    args: argparse.Namespace, video_url: str, *, stream: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.sglang_model,
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
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def request_sglang_once(
    args: argparse.Namespace, video_url: str, run_index: int, warmup: bool
) -> dict[str, Any]:
    if args.sglang_stream:
        return request_sglang_stream_once(args, video_url, run_index, warmup)

    payload = build_sglang_payload(args, video_url)
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{args.sglang_base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=args.request_timeout,
        )
        elapsed_s = time.perf_counter() - start
        result = response.json()
    except Exception as exc:
        elapsed_s = time.perf_counter() - start
        return {
            "row_type": "request",
            "backend": "sglang_mlx",
            "ok": False,
            "warmup": warmup,
            "run_index": run_index,
            "video_url": video_url,
            "elapsed_s": elapsed_s,
            "timings_s": {"request_total_s": elapsed_s},
            "error": f"{type(exc).__name__}: {exc}",
        }

    usage = result.get("usage") or {}
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    row = {
        "row_type": "request",
        "backend": "sglang_mlx",
        "ok": response.ok,
        "status_code": response.status_code,
        "warmup": warmup,
        "run_index": run_index,
        "video_url": video_url,
        "elapsed_s": elapsed_s,
        "timings_s": {"request_total_s": elapsed_s},
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


def request_sglang_stream_once(
    args: argparse.Namespace, video_url: str, run_index: int, warmup: bool
) -> dict[str, Any]:
    payload = build_sglang_payload(args, video_url, stream=True)
    start = time.perf_counter()
    first_content_s = None
    text_parts = []
    usage: dict[str, Any] = {}
    status_code = None
    try:
        response = requests.post(
            f"{args.sglang_base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=args.request_timeout,
            stream=True,
        )
        status_code = response.status_code
        if not response.ok:
            elapsed_s = time.perf_counter() - start
            return {
                "row_type": "request",
                "backend": "sglang_mlx",
                "ok": False,
                "status_code": status_code,
                "warmup": warmup,
                "run_index": run_index,
                "video_url": video_url,
                "elapsed_s": elapsed_s,
                "timings_s": {"request_total_s": elapsed_s},
                "error": response.text[:1000],
            }

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                if first_content_s is None:
                    first_content_s = time.perf_counter() - start
                text_parts.append(content)
        elapsed_s = time.perf_counter() - start
    except Exception as exc:
        elapsed_s = time.perf_counter() - start
        return {
            "row_type": "request",
            "backend": "sglang_mlx",
            "ok": False,
            "status_code": status_code,
            "warmup": warmup,
            "run_index": run_index,
            "video_url": video_url,
            "elapsed_s": elapsed_s,
            "timings_s": {"request_total_s": elapsed_s},
            "error": f"{type(exc).__name__}: {exc}",
        }

    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    stream_decode_s = (
        elapsed_s - first_content_s if first_content_s is not None else None
    )
    timings_s = {"request_total_s": elapsed_s}
    if first_content_s is not None:
        timings_s["ttft_s"] = first_content_s
    if stream_decode_s is not None:
        timings_s["stream_decode_s"] = stream_decode_s

    row = {
        "row_type": "request",
        "backend": "sglang_mlx",
        "ok": True,
        "status_code": status_code,
        "stream": True,
        "warmup": warmup,
        "run_index": run_index,
        "video_url": video_url,
        "elapsed_s": elapsed_s,
        "timings_s": timings_s,
        "ttft_s": first_content_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_s": (
            completion_tokens / elapsed_s if completion_tokens and elapsed_s else None
        ),
        "total_tokens_per_s": total_tokens / elapsed_s if total_tokens and elapsed_s else None,
        "text": "".join(text_parts),
    }
    if completion_tokens and stream_decode_s is not None and completion_tokens > 1:
        row["post_ttft_tokens_per_s"] = (completion_tokens - 1) / stream_decode_s
    return row


def parse_json_from_stdout(stdout: str) -> Any:
    stripped = stdout.strip()
    if not stripped:
        return None
    for candidate in reversed(stripped.splitlines()):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def request_custom_command_once(
    args: argparse.Namespace, video_url: str, run_index: int, warmup: bool
) -> dict[str, Any]:
    values = {
        "model_path": args.custom_model_path,
        "video_url": video_url,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "run_index": run_index,
    }
    cmd = format_command(args.custom_command, values)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=args.request_timeout,
            check=False,
        )
        elapsed_s = time.perf_counter() - start
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - start
        row = custom_base_row(args, video_url, run_index, warmup, elapsed_s)
        row.update(
            {
                "ok": False,
                "command": cmd,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
                "error": f"TimeoutExpired after {args.request_timeout}s",
            }
        )
        return row
    parsed = parse_json_from_stdout(proc.stdout)
    row = custom_base_row(args, video_url, run_index, warmup, elapsed_s)
    row.update(
        {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    )
    merge_custom_result(row, parsed)
    if proc.returncode != 0:
        row["error"] = proc.stderr.strip() or proc.stdout.strip()
    return row


def request_custom_callable_once(
    args: argparse.Namespace, video_url: str, run_index: int, warmup: bool
) -> dict[str, Any]:
    module_name, func_name = args.custom_callable.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    start = time.perf_counter()
    try:
        result = func(
            model_path=args.custom_model_path,
            video_url=video_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        elapsed_s = time.perf_counter() - start
        row = custom_base_row(args, video_url, run_index, warmup, elapsed_s)
    except Exception as exc:
        elapsed_s = time.perf_counter() - start
        row = custom_base_row(args, video_url, run_index, warmup, elapsed_s)
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["ok"] = True
    merge_custom_result(row, result)
    return row


def custom_base_row(
    args: argparse.Namespace,
    video_url: str,
    run_index: int,
    warmup: bool,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "row_type": "request",
        "backend": "custom_mlx_hybrid",
        "ok": False,
        "warmup": warmup,
        "run_index": run_index,
        "video_url": video_url,
        "model_path": args.custom_model_path,
        "elapsed_s": elapsed_s,
    }


def merge_custom_result(row: dict[str, Any], result: Any) -> None:
    if isinstance(result, dict):
        row["result"] = result
        text = result.get("text") or result.get("output") or result.get("content")
        usage = result.get("usage") or {}
        completion_tokens = result.get("completion_tokens") or usage.get(
            "completion_tokens"
        )
        prompt_tokens = result.get("prompt_tokens") or usage.get("prompt_tokens")
        total_tokens = result.get("total_tokens") or usage.get("total_tokens")
        if text is not None:
            row["text"] = text
        if prompt_tokens is not None:
            row["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            row["completion_tokens"] = completion_tokens
            row["completion_tokens_per_s"] = completion_tokens / row["elapsed_s"]
        if total_tokens is not None:
            row["total_tokens"] = total_tokens
            row["total_tokens_per_s"] = total_tokens / row["elapsed_s"]
        timings = result.get("timings_s")
        if isinstance(timings, dict):
            row["timings_s"] = timings
            ttft_s = result.get("time_to_first_token_s")
            if ttft_s is None:
                ttft_s = timings.get("time_to_first_token_s")
            decode_s = timings.get("mlx_decode_s") or timings.get("decode_s")
            if ttft_s is not None:
                row["ttft_s"] = ttft_s
            if decode_s is not None:
                row["decode_s"] = decode_s
            if completion_tokens is not None and decode_s:
                row["decode_tokens_per_s"] = completion_tokens / decode_s
        if result.get("decode_tokens_per_s") is not None:
            row["decode_tokens_per_s"] = result["decode_tokens_per_s"]
        if "ok" in result:
            row["ok"] = bool(result["ok"])
    elif result is not None:
        row["text"] = str(result)
        row["result"] = result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    backends = sorted({row.get("backend", "unknown") for row in rows})
    for backend in backends:
        measured = [
            row
            for row in rows
            if row.get("backend") == backend and row.get("ok") and not row.get("warmup")
            and row.get("row_type", "request") == "request"
        ]
        summary[backend] = summarize_rows(measured)
        startup_rows = [
            row
            for row in rows
            if row.get("backend") == backend
            and row.get("ok")
            and row.get("row_type") == "sglang_server_start"
        ]
        if startup_rows:
            summary[backend]["server_startup_s"] = summarize_metric(
                row["elapsed_s"] for row in startup_rows if row.get("elapsed_s")
            )
    if "sglang_mlx" in summary and "custom_mlx_hybrid" in summary:
        sglang_elapsed = (summary["sglang_mlx"].get("elapsed_s") or {}).get("mean")
        custom_elapsed = (summary["custom_mlx_hybrid"].get("elapsed_s") or {}).get(
            "mean"
        )
        if sglang_elapsed and custom_elapsed:
            summary["comparison"] = {
                "sglang_mlx_vs_custom_mlx_hybrid_elapsed_mean_ratio": (
                    sglang_elapsed / custom_elapsed
                ),
                "custom_mlx_hybrid_vs_sglang_mlx_elapsed_mean_ratio": (
                    custom_elapsed / sglang_elapsed
                ),
            }
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_measured": 0}

    def metric(name: str) -> dict[str, float] | None:
        return summarize_metric(row[name] for row in rows if row.get(name) is not None)

    timing_names = sorted(
        {
            name
            for row in rows
            for name in (row.get("timings_s") or {})
            if row.get("timings_s", {}).get(name) is not None
        }
    )

    return {
        "num_measured": len(rows),
        "elapsed_s": metric("elapsed_s"),
        "ttft_s": metric("ttft_s"),
        "decode_s": metric("decode_s"),
        "completion_tokens_per_s": metric("completion_tokens_per_s"),
        "decode_tokens_per_s": metric("decode_tokens_per_s"),
        "post_ttft_tokens_per_s": metric("post_ttft_tokens_per_s"),
        "total_tokens_per_s": metric("total_tokens_per_s"),
        "timings_s": {
            name: summarize_metric(
                row["timings_s"][name]
                for row in rows
                if row.get("timings_s", {}).get(name) is not None
            )
            for name in timing_names
        },
    }


def summarize_metric(values: Any) -> dict[str, float] | None:
    vals = list(values)
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def validate_custom_args(args: argparse.Namespace) -> None:
    if args.mode not in {"custom-mlx", "both"}:
        return
    if bool(args.custom_command) == bool(args.custom_callable):
        raise ValueError("Pass exactly one of --custom-command or --custom-callable.")
    if args.custom_command:
        valid_fields = {
            "model_path",
            "video_url",
            "prompt",
            "max_tokens",
            "temperature",
            "run_index",
        }
        fields = {
            name
            for _, name, _, _ in Formatter().parse(args.custom_command)
            if name is not None
        }
        unknown = fields - valid_fields
        if unknown:
            raise ValueError(f"Unknown --custom-command placeholders: {sorted(unknown)}")


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_custom_args(args)
    videos = load_video_urls(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    server_proc = None
    try:
        with output_path.open("a", encoding="utf-8") as f:
            if args.mode in {"sglang-mlx", "both"} and args.launch_sglang:
                cmd = build_sglang_command(args)
                env = os.environ.copy()
                env["SGLANG_USE_MLX"] = "1"
                if args.sglang_use_custom_rope:
                    env["SGLANG_MLX_USE_CUSTOM_ROPE"] = "1"
                print("starting SGLang MLX server:")
                print(" ".join(shlex.quote(part) for part in cmd))
                start = time.perf_counter()
                server_proc = subprocess.Popen(cmd, env=env, text=True)
                try:
                    wait_for_sglang(args.sglang_base_url, args.server_ready_timeout)
                    elapsed_s = time.perf_counter() - start
                    row = {
                        "row_type": "sglang_server_start",
                        "backend": "sglang_mlx",
                        "ok": True,
                        "elapsed_s": elapsed_s,
                        "command": cmd,
                        "base_url": args.sglang_base_url,
                    }
                except Exception as exc:
                    elapsed_s = time.perf_counter() - start
                    row = {
                        "row_type": "sglang_server_start",
                        "backend": "sglang_mlx",
                        "ok": False,
                        "elapsed_s": elapsed_s,
                        "command": cmd,
                        "base_url": args.sglang_base_url,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    write_row(f, row)
                    rows.append(row)
                    print_row(row)
                    raise
                write_row(f, row)
                rows.append(row)
                print_row(row)

            for video_index, video_url in enumerate(videos):
                for run_index in range(args.warmup + args.runs):
                    warmup = run_index < args.warmup
                    if args.mode in {"sglang-mlx", "both"}:
                        row = request_sglang_once(args, video_url, run_index, warmup)
                        row["video_index"] = video_index
                        write_row(f, row)
                        rows.append(row)
                        print_row(row)
                    if args.mode in {"custom-mlx", "both"}:
                        if args.custom_command:
                            row = request_custom_command_once(
                                args, video_url, run_index, warmup
                            )
                        else:
                            row = request_custom_callable_once(
                                args, video_url, run_index, warmup
                            )
                        row["video_index"] = video_index
                        write_row(f, row)
                        rows.append(row)
                        print_row(row)
    finally:
        if args.stop_sglang:
            stop_process(server_proc)

    return {"summary": summarize(rows), "output": str(output_path)}


def write_row(f, row: dict[str, Any]) -> None:
    row["created_at"] = datetime.now(timezone.utc).isoformat()
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
    f.flush()


def print_row(row: dict[str, Any]) -> None:
    if row.get("row_type") == "sglang_server_start":
        status = "ok" if row.get("ok") else "failed"
        print(
            f"event backend=sglang_mlx type=server_start {status} "
            f"elapsed={row['elapsed_s']:.3f}s"
        )
        return

    label = "warmup" if row.get("warmup") else "run"
    backend = row.get("backend", "unknown")
    if row.get("ok"):
        tps = row.get("completion_tokens_per_s")
        tps_str = f"{tps:.2f}" if tps is not None else "n/a"
        ttft = row.get("ttft_s")
        ttft_str = f" ttft={ttft:.3f}s" if ttft is not None else ""
        print(
            f"{label} backend={backend} video={row.get('video_index')} "
            f"idx={row.get('run_index')} elapsed={row['elapsed_s']:.3f}s "
            f"completion_tps={tps_str}{ttft_str}"
        )
    else:
        print(
            f"{label} backend={backend} video={row.get('video_index')} "
            f"idx={row.get('run_index')} failed elapsed={row['elapsed_s']:.3f}s "
            f"error={row.get('error')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Marlin on SGLang MLX and custom MLX hybrid."
    )
    parser.add_argument(
        "--mode",
        choices=["sglang-mlx", "custom-mlx", "both"],
        default="both",
        help="Which benchmark path to run.",
    )
    parser.add_argument("--video-url", action="append", default=[])
    parser.add_argument("--video-list")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument(
        "--output",
        default="benchmark/marlin_video/mlx_compare_results.jsonl",
        help="JSONL output path.",
    )

    parser.add_argument("--sglang-model-path", default="junwatu/Marlin-2B-MLX-8bit")
    parser.add_argument(
        "--sglang-tokenizer-path",
        help=(
            "Optional tokenizer/processor repo for SGLang. For Marlin MLX "
            "8-bit weights, use NemoStation/Marlin-2B."
        ),
    )
    parser.add_argument("--sglang-model", default="default")
    parser.add_argument("--sglang-host", default="0.0.0.0")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--sglang-base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--sglang-quantization", choices=["mlx_q4", "mlx_q8"])
    parser.add_argument("--mm-process-config", default=DEFAULT_MM_PROCESS_CONFIG)
    parser.add_argument("--server-ready-timeout", type=float, default=900)
    parser.add_argument("--sglang-extra-arg", action="append")
    parser.add_argument(
        "--sglang-server-command",
        help=(
            "Override server launch command. Supports placeholders: {model_path}, "
            "{tokenizer_path}, {served_model_name}, {host}, {port}, "
            "{mm_process_config}."
        ),
    )
    parser.add_argument(
        "--no-launch-sglang",
        dest="launch_sglang",
        action="store_false",
        help="Use an already-running SGLang server.",
    )
    parser.add_argument(
        "--no-stop-sglang",
        dest="stop_sglang",
        action="store_false",
        help="Leave a launched SGLang server running.",
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Launch SGLang MLX with synchronous scheduler behavior.",
    )
    parser.add_argument(
        "--sglang-use-custom-rope",
        action="store_true",
        help="Set SGLANG_MLX_USE_CUSTOM_ROPE=1 for the SGLang run.",
    )
    parser.add_argument(
        "--sglang-stream",
        action="store_true",
        help=(
            "Use streaming chat completions for SGLang requests and record TTFT "
            "when the server emits token deltas."
        ),
    )

    parser.add_argument("--custom-model-path", default="junwatu/Marlin-2B-MLX-8bit")
    parser.add_argument(
        "--custom-command",
        help=(
            "Command template for the custom MLX hybrid runner. Supports "
            "{model_path}, {video_url}, {prompt}, {max_tokens}, {temperature}, "
            "and {run_index}. If stdout ends with JSON containing text/usage fields, "
            "those are recorded."
        ),
    )
    parser.add_argument(
        "--custom-callable",
        help=(
            "Python callable as module:function. The function receives model_path, "
            "video_url, prompt, max_tokens, and temperature keyword arguments."
        ),
    )

    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
