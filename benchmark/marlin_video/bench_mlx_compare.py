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


def request_sglang_once(
    args: argparse.Namespace, video_url: str, run_index: int, warmup: bool
) -> dict[str, Any]:
    payload = {
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
            "backend": "sglang_mlx",
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
        "backend": "sglang_mlx",
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
        total_tokens = result.get("total_tokens") or usage.get("total_tokens")
        if text is not None:
            row["text"] = text
        if completion_tokens is not None:
            row["completion_tokens"] = completion_tokens
            row["completion_tokens_per_s"] = completion_tokens / row["elapsed_s"]
        if total_tokens is not None:
            row["total_tokens"] = total_tokens
            row["total_tokens_per_s"] = total_tokens / row["elapsed_s"]
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
        ]
        summary[backend] = summarize_rows(measured)
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
        vals = [row[name] for row in rows if row.get(name) is not None]
        if not vals:
            return None
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "num_measured": len(rows),
        "elapsed_s": metric("elapsed_s"),
        "completion_tokens_per_s": metric("completion_tokens_per_s"),
        "total_tokens_per_s": metric("total_tokens_per_s"),
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
        if args.mode in {"sglang-mlx", "both"} and args.launch_sglang:
            cmd = build_sglang_command(args)
            env = os.environ.copy()
            env["SGLANG_USE_MLX"] = "1"
            if args.sglang_use_custom_rope:
                env["SGLANG_MLX_USE_CUSTOM_ROPE"] = "1"
            print("starting SGLang MLX server:")
            print(" ".join(shlex.quote(part) for part in cmd))
            server_proc = subprocess.Popen(cmd, env=env, text=True)
            wait_for_sglang(args.sglang_base_url, args.server_ready_timeout)

        with output_path.open("a", encoding="utf-8") as f:
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
    label = "warmup" if row.get("warmup") else "run"
    backend = row.get("backend", "unknown")
    if row.get("ok"):
        tps = row.get("completion_tokens_per_s")
        tps_str = f"{tps:.2f}" if tps is not None else "n/a"
        print(
            f"{label} backend={backend} video={row.get('video_index')} "
            f"idx={row.get('run_index')} elapsed={row['elapsed_s']:.3f}s "
            f"completion_tps={tps_str}"
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
            "{served_model_name}, {host}, {port}, {mm_process_config}."
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
