# Marlin-2B SGLang Video Benchmark

This benchmark is for running `NemoStation/Marlin-2B` through SGLang's OpenAI-compatible server and comparing it with the custom MLX 8-bit results from `junwatu/Marlin-2B-MLX-8bit`.

For Apple Silicon SGLang MLX vs custom MLX hybrid instructions, see [README_MLX.md](README_MLX.md).

The model is Marlin. The SGLang execution class override below is only needed because the checkpoint advertises `MarlinForConditionalGeneration`, while this SGLang fork can execute it through the native Qwen3.5 implementation.

## Environment

Use a larger NVIDIA GPU than a 6 GB RTX 4050. On that GPU, the BF16 weights alone used 4.31 GiB, and the recommended video preprocessing path OOMed while allocating another 78 MiB.

The real Marlin processor requires Transformers 5.7.0 or newer:

```bash
python -m pip install --upgrade "transformers>=5.7.0" "huggingface-hub>=1.18.0"
python -m pip install --upgrade "qwen-vl-utils>=0.0.14" torchcodec av requests
```

If SGLang refuses to start because of the Torch/CuDNN compatibility guard, set `SGLANG_DISABLE_CUDNN_CHECK=1` for the benchmark run and record that in the results.

## Launch SGLang

This uses the Marlin model card's video scale: `fps=2`, `min_frames=4`, `max_frames=240`, `max_pixels=200704`.

```bash
python -m sglang.launch_server \
  --model-path NemoStation/Marlin-2B \
  --served-model-name default \
  --trust-remote-code \
  --enable-multimodal \
  --dtype bfloat16 \
  --context-length 4096 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 30000 \
  --mm-process-config '{"video":{"fps":2.0,"min_frames":4,"max_frames":240,"max_pixels":200704}}' \
  --json-model-override-args '{"architectures":["Qwen3_5ForConditionalGeneration"]}'
```

If the server OOMs during startup, retry with `--disable-cuda-graph` or reduce `--max-total-tokens`. If it OOMs during video preprocessing, the GPU is still too tight for a meaningful comparison with the MLX benchmark.

## Run The Client

For an apples-to-apples comparison, pass the same 8-second clips used for the MLX run.

```bash
python benchmark/marlin_video/bench_sglang.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model default \
  --video-url https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4 \
  --max-tokens 384 \
  --warmup 1 \
  --runs 5
```

For multiple clips:

```bash
python benchmark/marlin_video/bench_sglang.py \
  --video-list /path/to/video_urls.txt \
  --max-tokens 384 \
  --warmup 1 \
  --runs 5
```

The client writes JSONL rows with wall time, token usage, throughput, response text, and any server error. Treat first-request timings separately because video download, preprocessing cache misses, and CUDA graph capture can dominate them.

## Comparison Notes

The MLX model card reports roughly 2.5 GB model size, around 5 GB peak memory, and about 28 seconds per 8-second video in hybrid mode with correct timestamps. The SGLang path here keeps BF16 weights and pays CUDA server memory overhead, so the first question to answer on a bigger GPU is not just tokens/sec. Check whether the full recommended video preprocessing path fits comfortably, then compare steady-state latency over the same clips and prompt.
