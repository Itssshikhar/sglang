# Marlin-2B Video Captioning on Apple Silicon: SGLang MLX vs. Custom MLX Hybrid

Benchmark report, 2026-06-11.

## TL;DR

| | SGLang MLX server | Custom MLX hybrid |
|---|---|---|
| Mean elapsed / video | **9.38 s** | **67.8 s** |
| Completion tokens/s | 19.6 | 4.1 |
| Caption correctness | ❌ **hallucinated** (vision features silently dropped) | ✅ accurate, with temporal events |

**Headline finding: SGLang's MLX backend has no multimodal support.** Video
frames are downloaded, decoded, and preprocessed, then discarded before the
model forward pass — the model sees only text tokens and hallucinates a
caption. Its 7× speed advantage is therefore meaningless for video
captioning. The custom hybrid (HF transformers preprocessing + mlx-vlm
generation) is the only path that actually captions video on this stack.

Evidence: `python/sglang/srt/hardware_backend/mlx/model_runner.py` embeds
token ids via `embed_tokens` only; there is no `input_embeddings` injection or
mm-feature handling anywhere under `python/sglang/srt/hardware_backend/mlx/`,
and `mlx_lm`'s `qwen3_5` wrapper explicitly skips `vision_tower`/`model.visual`
weights at load time.

## Test setup

| Component | Value |
|---|---|
| Hardware | Apple M4, 16 GB RAM, macOS 26.2 |
| Toolchain | Xcode 26.4 + Metal Toolchain 17E188 (separate download, see below) |
| Python | 3.12.11 (uv venv `sglang-metal`) |
| sglang | upstream `main` @ `73d0989d9` + this benchmark dir + `patches/sglang_main_mps_mlx_fixes.patch` |
| mlx / mlx-lm / mlx-vlm | 0.31.2 / 0.31.3 / 0.6.3 |
| transformers / torch / torchcodec | 5.11.0 / 2.11.0 / 0.14.0 |
| Video | `jobs_presenting_ipod.mp4` (sgl-test-files; Steve Jobs presenting an iPod on stage) |
| Prompt | benchmark default (spatial description + time-ranged events) |
| Sampling | temperature 0, max_tokens 384, 1 warmup + 3 measured runs |

Models:

- SGLang side: `lunahr/Marlin-2B-ungated` (BF16) with `--quantization mlx_q8`
  (load-time 8-bit, 3.51 GB → 1.86 GB weights). This is a community re-upload
  of the gated `NemoStation/Marlin-2B`; provenance is unverified. It was used
  because the run predated HF access to the gated repos. With access, prefer
  `junwatu/Marlin-2B-MLX-8bit` directly.
- Custom side: `junwatu/Marlin-2B-MLX-8bit` (pre-quantized MLX) for vision +
  generation, plus `NemoStation/Marlin-2B` (BF16, HF) for input preparation.

## Results

### SGLang MLX (server path)

```
warmup  9.771 s   18.8 tok/s   184 completion tokens
run 1   9.624 s   19.1 tok/s   184
run 2   9.124 s   20.2 tok/s   184
run 3   9.391 s   19.6 tok/s   184
mean    9.38 s    19.6 tok/s
```

Sample output (deterministic, temperature 0) — **wrong content**:

> "Scene: The video presents a static, high-angle shot of a large, open body
> of water… There are no people, animals, or man-made structures visible…"

### Custom MLX hybrid

```
warmup  69.585 s   3.98 tok/s   277 completion tokens
run 1   74.016 s   3.74 tok/s   277
run 2   66.749 s   4.15 tok/s   277
run 3   62.649 s   4.42 tok/s   277
mean    67.8 s     4.1 tok/s
```

Sample output — **correct content**:

> "Scene: A man in a black t-shirt and dark jeans stands on a stage,
> presenting a product… He is holding a small, silver, rectangular device,
> which he identifies as an iPod Nano…
> Events: \<0.0 - 1.0\> The man stands still on the stage. \<1.0 - 4.5\> The
> man gestures with his right hand while speaking. …"

### Fairness caveats

- The comparison is structurally lopsided on speed: SGLang serves from a
  warm, persistent server, while each custom run is a **cold process** that
  re-loads the 5.1 GB BF16 HF checkpoint and the 2.5 GB MLX checkpoint
  (the model card claims ~28 s/video with models resident). A persistent
  runner (e.g. the model card's FastAPI pattern) would exclude load time.
- The custom runner decodes greedily one token at a time in Python — no
  batched decode, no overlap scheduling.
- The two paths used different checkpoints (`lunahr` mirror + load-time q8
  vs. `junwatu` pre-quantized q8). Token counts also differ (184 vs. 277)
  because the SGLang side, being blind, terminates earlier.
- None of this changes the correctness conclusion: SGLang MLX output is
  hallucinated regardless of speed.

## How to reproduce

Follow `README_MLX.md` for environment setup, with these corrections:

1. **Xcode 26 ships without the Metal shader compiler.** `xcode-select
   --install` is not sufficient and `setup_metal.py` fails with "Apple Metal
   shader compiler not found". Fix:
   `xcodebuild -downloadComponent MetalToolchain` (~690 MB).
2. **`uv venv` creates a venv without pip.** Run `python -m ensurepip` and
   `pip install -U pip setuptools wheel` before `setup_metal.py install`.
3. Install the video decoder: `uv pip install torchcodec` (sglang prefers it;
   `decord`/`eva-decord` have no macOS arm64 wheels).
4. **This benchmark requires upstream sglang `main`** (≥ `73d0989d9`); this
   branch's base predates the MLX backend entirely (no `setup_metal.py`, no
   `SGLANG_USE_MLX`). Overlay this directory onto a main checkout, then apply
   `git apply benchmark/marlin_video/patches/sglang_main_mps_mlx_fixes.patch`.
5. Apply `patches/mlx_lm_qwen3_5_tied_lm_head.patch` to the venv's `mlx_lm`.

SGLang side (the architecture override is required because Marlin's config
declares the custom-code arch `MarlinForConditionalGeneration`, which sglang
cannot resolve; Marlin is `model_type=qwen3_5`, natively supported):

```bash
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode sglang-mlx \
  --sglang-model-path junwatu/Marlin-2B-MLX-8bit \
  --sglang-extra-arg="--json-model-override-args '{\"architectures\":[\"Qwen3_5ForConditionalGeneration\"]}'" \
  --sglang-extra-arg="--disable-radix-cache" \
  --warmup 1 --runs 3
```

Custom side:

```bash
python benchmark/marlin_video/bench_mlx_compare.py \
  --mode custom-mlx \
  --custom-model-path junwatu/Marlin-2B-MLX-8bit \
  --custom-command 'python benchmark/marlin_video/marlin_mlx_hybrid.py --model {model_path} --video-url {video_url} --prompt {prompt} --max-tokens {max_tokens} --temperature {temperature}' \
  --max-tokens 384 --warmup 1 --runs 3
```

Both `NemoStation/Marlin-2B` and `junwatu/Marlin-2B-MLX-8bit` are gated
(auto-approve): `hf auth login` and accept the conditions on each model page.

## Failure log

Every failure hit while bringing this up, in order, with root cause and fix.
Fixes marked **[repo]** are files in this directory; **[patch]** are in
`patches/sglang_main_mps_mlx_fixes.patch` (apply on upstream main); **[venv]**
patch installed packages (see `patches/`); **[flag]** are runtime workarounds;
**[env]** are environment setup steps.

| # | Failure | Root cause | Fix |
|---|---|---|---|
| 1 | `setup_metal.py`: "Apple Metal shader compiler not found" despite full Xcode | Xcode 26 ships the Metal toolchain as a separate downloadable component | `xcodebuild -downloadComponent MetalToolchain` **[env]** |
| 2 | `ValueError: Cannot find model module. 'MarlinForConditionalGeneration' is not a registered model…` at server start | Marlin's config declares a custom-code arch; its `auto_map` has only `AutoModelForCausalLM`, and sglang's gate requires `AutoModel` | `--json-model-override-args '{"architectures":["Qwen3_5ForConditionalGeneration"]}'` (Marlin is Qwen3.5-based) **[flag]** |
| 3 | `json.decoder.JSONDecodeError` parsing the override args | `bench_mlx_compare.py` runs `shlex.split()` on `--sglang-extra-arg` values, which strips the JSON's double quotes | Wrap the JSON in single quotes inside the extra-arg string (see repro commands) **[flag]** |
| 4 | `ValueError: Received 1 parameters not in model: language_model.lm_head.weight` during MLX weight load | mlx_lm qwen3_5 VL wrapper prefixes checkpoint keys with `language_model.` before the tied-embeddings drop, which only pops the un-prefixed `lm_head.weight` | `patches/mlx_lm_qwen3_5_tied_lm_head.patch` **[venv]** |
| 5 | `AttributeError: 'MlxAuxiliaryStateReqToTokenPool' object has no attribute 'mamba_allocator'` in the scheduler idle loop | Pool stats observer assumes the CUDA hybrid-SSM pool API; the MLX pool tracks SSM state internally | Guard in `pool_stats_observer.py` **[patch]** |
| 6 | `TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'` in the idle loop (after #5) | Invariant checker runs the mamba-pool check on stats that now carry no mamba fields | Guard in `invariant_checker.py` **[patch]** |
| 7 | Benchmark client stuck polling `/v1/model_info` (404) until timeout, server already healthy | Upstream sglang serves model info at `/model_info` on the root app, not under `/v1` | Readiness probe tries `/model_info`, `/get_model_info`, `/health` (`bench_mlx_compare.py`) **[repo]** |
| 8 | `RuntimeError: … No module named 'decord'` on every video request | No video decoder installed; `decord`/`eva-decord` ship no macOS arm64 wheels | `pip install torchcodec` (sglang's preferred backend) **[env]** |
| 9 | `RuntimeError: Attempted to set the storage of a tensor on device "cpu" to a storage on different device "mps:0"` during video preprocessing | `pin_memory()` on CPU tensors is broken when MPS is torch's accelerator; pinning only benefits CUDA H2D copies | Pin only when CUDA is available (`video_decoder.py`, `qwen_vl.py`) **[patch]** |
| 10 | `TypeError: merged_typed_dict.__init__() got an unexpected keyword argument 'max_pixels'` in the HF processor call | sglang forwards the whole `--mm-process-config` video dict into `videos_kwargs`; transformers ≥5 strictly validates typed kwargs, and the sampling/resize keys were already consumed by sglang's own `preprocess_video` | Filter consumed keys before forwarding; also stop defaulting the fast image processor to `cuda:0` on non-CUDA hosts (`base_processor.py`) **[patch]** |
| 11 | Scheduler crash in `unified_cache_components/mamba_component.py` (`mamba_allocator` again) on the **second** request — first request succeeded | Radix prefix-cache matching for hybrid-SSM models requires the mamba allocator the MLX pool doesn't implement; only triggers once the tree has a cached prefix | `--disable-radix-cache` (also makes benchmark runs independent) **[flag]** |
| 12 | `argparse: argument --sglang-extra-arg: expected one argument` | argparse treats a dash-leading value as an option | Use `--sglang-extra-arg=--disable-radix-cache` (equals form) **[flag]** |
| 13 | Custom runner: `ValueError: [max] Cannot max reduce zero size array` inside `mlx_vlm` `get_input_embeddings` | mlx_vlm 0.6.3's qwen3_5 `get_rope_index` mis-segments Marlin's interleaved timestamp/video token layout | Runner bypasses the internal rope call (embed + vision-merge manually) and injects HF-computed position ids — which is the point of the hybrid recipe anyway (`marlin_mlx_hybrid.py`) **[repo]** |
| 14 | **Semantic failure**: SGLang MLX pipeline "works" end-to-end but captions describe the wrong video | SGLang's MLX backend drops multimodal features entirely (see TL;DR) | None available — requires implementing multimodal embedding injection in the MLX backend. Tracked as the main open issue. **[open]** |

Two non-bugs worth knowing: the fork branch this benchmark originated on was
cut from a base without any MLX/Metal support (no `setup_metal.py`, no
`SGLANG_USE_MLX` in the package), so the benchmark must be run on top of a
recent upstream `main`; and the `--quantization mlx_q8` flag is correctly
reported as a no-op when loading an already-quantized MLX repo.

## Files in this directory

| File | Purpose |
|---|---|
| `README_MLX.md` | Original setup/usage guide (see errata note at top) |
| `MLX_BENCHMARK_REPORT.md` | This report |
| `bench_mlx_compare.py` | Comparison harness (includes readiness-probe fix) |
| `marlin_mlx_hybrid.py` | Custom MLX hybrid runner (model-card recipe, CLI adapter for `--custom-command`) |
| `patches/sglang_main_mps_mlx_fixes.patch` | Five MPS/MLX fixes to apply on upstream sglang main |
| `patches/mlx_lm_qwen3_5_tied_lm_head.patch` | Required venv patch for mlx_lm |
| `bench_sglang.py`, `README.md` | Pre-existing CUDA-path benchmark (untouched) |

Raw per-run rows (including all failed attempts above) are appended to
`mlx_compare_results.jsonl`, which is git-ignored (`*.jsonl`).

## Open follow-ups

1. **SGLang MLX multimodal support** — the blocker for any real SGLang-side
   number. Until then, `sglang-mlx` mode measures text throughput through a
   video-shaped request, not captioning.
2. Upstream the five sglang MPS/hybrid-SSM fixes in
   `patches/sglang_main_mps_mlx_fixes.patch`.
3. Report the mlx_vlm qwen3_5 `get_rope_index` crash (failure #13) upstream.
4. Persistent-process custom runner to measure warm generation speed
   (model card claims ~28 s/video vs. our 67.8 s cold).
5. Re-run the SGLang side with the canonical `junwatu/Marlin-2B-MLX-8bit`
   instead of the unverified `lunahr` mirror (access now available).
