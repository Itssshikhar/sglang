# Marlin MLX Comparison Benchmark

> **Errata (2026-06-11):** see [MLX_BENCHMARK_REPORT.md](MLX_BENCHMARK_REPORT.md)
> for actual results and required corrections to these instructions. Notably:
> this benchmark must run on upstream sglang `main` (this branch's base has no
> MLX backend) with `patches/sglang_main_mps_mlx_fixes.patch` applied, and the
> SGLang MLX path currently **drops video features entirely** — its captions
> are hallucinated. A working custom runner is provided in
> `marlin_mlx_hybrid.py`.

This benchmark compares two Apple Silicon paths for Marlin video captioning:

1. SGLang's native MLX runtime, launched with `SGLANG_USE_MLX=1`.
2. A custom MLX hybrid implementation, such as `junwatu/Marlin-2B-MLX-8bit`.

The SGLang side is benchmarked through the OpenAI-compatible server. The custom
MLX side is benchmarked as either a shell command template or a Python
`module:function` callable, because custom-code MLX repos do not expose one
stable CLI/API shape.

## Requirements

Run this on Apple Silicon macOS. The MLX backend does not run on Linux CUDA
machines.

Install Xcode Command Line Tools:

```bash
xcode-select --install
xcode-select -p
xcrun --find metal
```

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a Python 3.12 environment:

```bash
uv venv -p 3.12 sglang-metal
source sglang-metal/bin/activate
python -m pip install --upgrade pip
```

Install SGLang from this checkout with Apple/Metal support:

```bash
uv run sgl-kernel/setup_metal.py install
rm -f python/pyproject.toml
mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```

Install Marlin/video dependencies:

```bash
uv pip install --upgrade \
  "transformers>=5.7.0" \
  "huggingface-hub>=1.18.0" \
  "qwen-vl-utils>=0.0.14" \
  av requests
```

Install MLX libraries used by custom MLX runners:

```bash
uv pip install --upgrade mlx mlx-lm mlx-vlm
```

Authenticate with Hugging Face and request access to any gated model repos:

```bash
huggingface-cli login
```

You need access to:

- `NemoStation/Marlin-2B` if you test the original BF16 checkpoint.
- `junwatu/Marlin-2B-MLX-8bit` if you test the custom 8-bit MLX checkpoint.

## SGLang MLX Server Path

SGLang's Apple backend is selected with `SGLANG_USE_MLX=1`, not with a
`--backend` flag. A minimal manual launch is:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path junwatu/Marlin-2B-MLX-8bit \
  --served-model-name default \
  --trust-remote-code \
  --enable-multimodal \
  --disable-cuda-graph \
  --host 0.0.0.0 \
  --port 30000 \
  --mm-process-config '{"video":{"fps":2.0,"min_frames":4,"max_frames":240,"max_pixels":200704}}'
```

Useful SGLang MLX switches:

- `--disable-cuda-graph`: expected for Metal/MLX.
- `--disable-overlap-schedule`: use this for a synchronous comparison path.
- `SGLANG_MLX_USE_CUSTOM_ROPE=1`: opt into the custom Metal RoPE kernel.
- `--quantization mlx_q8`: quantize an fp16 model at load time. Do not use this
  if you are already loading an MLX 8-bit repo unless you want to test the flag's
  no-op behavior on pre-quantized weights.

## Custom MLX Hybrid Path

The comparison script supports two adapters.

Use `--custom-command` if your custom runner is a CLI:

```bash
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode custom-mlx \
  --custom-command 'python /path/to/marlin_mlx_hybrid.py --model {model_path} --video-url {video_url} --prompt {prompt} --max-tokens {max_tokens}' \
  --custom-model-path junwatu/Marlin-2B-MLX-8bit \
  --warmup 1 \
  --runs 5
```

The command template may use these placeholders:

- `{model_path}`
- `{video_url}`
- `{prompt}`
- `{max_tokens}`
- `{temperature}`
- `{run_index}`

If your command itself contains literal braces, escape them as `{{` and `}}`
because the template is expanded with Python string formatting.

If the command prints a final JSON line, the script records it. The most useful
shape is:

```json
{"ok": true, "text": "...", "usage": {"completion_tokens": 267, "total_tokens": 3518}}
```

Use `--custom-callable` if your custom runner exposes a Python function:

```bash
PYTHONPATH=/path/to/custom/marlin/repo:$PYTHONPATH \
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode custom-mlx \
  --custom-callable marlin_hybrid_bench:run_once \
  --custom-model-path junwatu/Marlin-2B-MLX-8bit \
  --warmup 1 \
  --runs 5
```

The callable must accept keyword arguments:

```python
def run_once(model_path, video_url, prompt, max_tokens, temperature):
    ...
    return {
        "ok": True,
        "text": generated_text,
        "usage": {
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
```

Token counts are optional. If they are missing, the summary still reports wall
time, but token/sec fields are omitted for the custom path.

## Run Both Paths

This launches SGLang MLX, waits for `/v1/model_info`, runs both backends on the
same input, writes JSONL rows, and stops the SGLang server at the end:

```bash
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode both \
  --sglang-model-path junwatu/Marlin-2B-MLX-8bit \
  --custom-model-path junwatu/Marlin-2B-MLX-8bit \
  --custom-command 'python /path/to/marlin_mlx_hybrid.py --model {model_path} --video-url {video_url} --prompt {prompt} --max-tokens {max_tokens}' \
  --video-url https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4 \
  --max-tokens 384 \
  --warmup 1 \
  --runs 5 \
  --output benchmark/marlin_video/mlx_compare_results.jsonl
```

For the synchronous SGLang MLX scheduler path:

```bash
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode both \
  --disable-overlap-schedule \
  --sglang-model-path junwatu/Marlin-2B-MLX-8bit \
  --custom-model-path junwatu/Marlin-2B-MLX-8bit \
  --custom-callable marlin_hybrid_bench:run_once \
  --warmup 1 \
  --runs 5
```

To use an already-running SGLang server:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server ... --port 30000

python benchmark/marlin_video/bench_mlx_compare.py \
  --mode both \
  --no-launch-sglang \
  --sglang-base-url http://127.0.0.1:30000/v1 \
  --custom-callable marlin_hybrid_bench:run_once
```

## Multiple Clips

Create a text file with one video URL per line:

```text
https://example.com/clip-1.mp4
https://example.com/clip-2.mp4
```

Then run:

```bash
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode both \
  --video-list /path/to/video_urls.txt \
  --custom-callable marlin_hybrid_bench:run_once \
  --warmup 1 \
  --runs 5
```

## Output

The script appends JSONL rows to:

```text
benchmark/marlin_video/mlx_compare_results.jsonl
```

Each row includes:

- `backend`: `sglang_mlx` or `custom_mlx_hybrid`
- `warmup`
- `run_index`
- `video_url`
- `elapsed_s`
- optional token counts and token/sec metrics
- generated text or raw custom command output

The final stdout summary groups measured, non-warmup rows by backend and includes
the mean elapsed-time ratio when both backends have successful measured runs.

## Notes For Fair Comparisons

- Use the same `video_url`, prompt, `max_tokens`, and temperature for both paths.
- Treat first-request latency separately. It can include model load, video
  download, preprocessing, kernel compilation, and cache misses.
- If you compare SGLang overlap scheduling against a synchronous custom hybrid
  runner, report that explicitly. Use `--disable-overlap-schedule` for a simpler
  sync-vs-sync comparison.
- Prefer the same quantized checkpoint for both paths. If SGLang MLX runs BF16
  while the custom path runs 8-bit, the result is useful but not apples-to-apples.
