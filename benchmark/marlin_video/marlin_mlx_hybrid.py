"""Marlin-2B custom MLX hybrid runner for bench_mlx_compare.py.

Implements the model card's recommended hybrid recipe
(https://huggingface.co/junwatu/Marlin-2B-MLX-8bit): HF transformers prepare
inputs and M-RoPE position ids, MLX runs generation. Prints a final JSON line
in the shape the benchmark records:
  {"ok": true, "text": "...", "usage": {"completion_tokens": N, "total_tokens": M}}
"""

import argparse
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("VIDEO_MAX_PIXELS", "200704")
os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "240")
os.environ.setdefault("FPS_MIN_FRAMES", "4")

HF_MODEL = "NemoStation/Marlin-2B"


def fetch_video(url: str) -> str:
    if os.path.exists(url):
        return url
    import requests

    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-model", default=HF_MODEL)
    ap.add_argument("--video-url", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    def mark(name: str, start: float) -> None:
        timings[name] = time.perf_counter() - start

    try:
        start = time.perf_counter()
        import mlx.core as mx
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from mlx_vlm import load as mlx_load
        from mlx_vlm.models.cache import make_prompt_cache

        mark("imports_s", start)

        start = time.perf_counter()
        video_path = fetch_video(args.video_url)
        mark("video_fetch_s", start)

        # 1. HF side: tokenize video + prompt, compute M-RoPE position ids.
        # bf16 instead of the card's fp32: get_rope_index only produces integer
        # position indices, weight dtype does not affect it, and fp32 would not
        # fit comfortably in 16 GB alongside the MLX model.
        start = time.perf_counter()
        hf_processor = AutoProcessor.from_pretrained(
            args.hf_model, trust_remote_code=True
        )
        mark("hf_processor_load_s", start)

        start = time.perf_counter()
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        mark("hf_model_load_s", start)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        start = time.perf_counter()
        inputs = hf_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        mark("hf_processor_apply_s", start)

        start = time.perf_counter()
        with torch.no_grad():
            position_ids, _ = hf_model.model.get_rope_index(
                input_ids=inputs["input_ids"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
                video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
            )
        mark("mrope_compute_s", start)

        start = time.perf_counter()
        del hf_model
        mark("hf_model_release_s", start)

        # 2. MLX side: vision encode + generate.
        start = time.perf_counter()
        mlx_model, mlx_processor = mlx_load(args.model)
        mark("mlx_model_load_s", start)

        start = time.perf_counter()
        input_ids = mx.array(inputs["input_ids"].numpy())
        pixel_values = mx.array(inputs["pixel_values_videos"].numpy())
        video_grid_thw = mx.array(inputs["video_grid_thw"].numpy())
        mx.eval(input_ids, pixel_values, video_grid_thw)
        mark("mlx_input_conversion_s", start)

        # Replicate get_input_embeddings minus its internal get_rope_index call,
        # which crashes on Marlin's interleaved timestamp/video token layout in
        # mlx_vlm 0.6.3. The HF-computed position ids are authoritative anyway.
        dtype = mlx_model.vision_tower.patch_embed.proj.weight.dtype

        start = time.perf_counter()
        inputs_embeds = mlx_model.language_model.model.embed_tokens(input_ids)
        mx.eval(inputs_embeds)
        mark("mlx_text_embedding_s", start)

        start = time.perf_counter()
        hidden_states, _ = mlx_model.vision_tower(
            pixel_values.astype(dtype), video_grid_thw
        )
        mx.eval(hidden_states)
        mark("mlx_vision_encode_s", start)

        start = time.perf_counter()
        inputs_embeds, _ = mlx_model.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            mlx_model.config.image_token_index,
            mlx_model.config.video_token_index,
        )
        mx.eval(inputs_embeds)
        mark("mlx_embedding_merge_s", start)

        class _Embeds:
            pass

        embedding_output = _Embeds()
        embedding_output.inputs_embeds = inputs_embeds
        mlx_model.language_model._position_ids = mx.array(position_ids.numpy())
        mlx_model.language_model._rope_deltas = None

        start = time.perf_counter()
        prompt_cache = make_prompt_cache(mlx_model.language_model)
        outputs = mlx_model.language_model(
            input_ids,
            inputs_embeds=embedding_output.inputs_embeds,
            cache=prompt_cache,
        )
        mx.eval(outputs.logits, [c.state for c in prompt_cache])
        mark("mlx_prefill_s", start)

        eos = mlx_model.config.eos_token_id
        eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

        start = time.perf_counter()
        y = mx.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        tokens = []
        time_to_first_token_s = None
        for _ in range(args.max_tokens):
            t = y.item()
            if t in eos_ids:
                break
            if time_to_first_token_s is None:
                time_to_first_token_s = time.perf_counter() - total_start
            tokens.append(t)
            if len(tokens) >= args.max_tokens:
                break
            outputs = mlx_model.language_model(y, cache=prompt_cache)
            y = mx.argmax(outputs.logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(y, [c.state for c in prompt_cache])
        mark("mlx_decode_s", start)

        start = time.perf_counter()
        text = mlx_processor.tokenizer.decode(tokens, skip_special_tokens=True)
        mark("token_decode_s", start)

        prompt_tokens = int(inputs["input_ids"].shape[1])
        timings["total_s"] = time.perf_counter() - total_start
        print(
            json.dumps(
                {
                    "ok": True,
                    "text": text,
                    "timings_s": timings,
                    "time_to_first_token_s": time_to_first_token_s,
                    "decode_tokens_per_s": (
                        len(tokens) / timings["mlx_decode_s"]
                        if timings["mlx_decode_s"] > 0
                        else None
                    ),
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(tokens),
                        "total_tokens": prompt_tokens + len(tokens),
                    },
                }
            )
        )
    except Exception as exc:  # noqa: BLE001 - benchmark wants a JSON error line
        import traceback

        timings["total_s"] = time.perf_counter() - total_start
        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timings_s": timings,
                }
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
