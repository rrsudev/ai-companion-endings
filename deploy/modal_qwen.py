"""Serve the interactor model (Qwen/Qwen3.5-9B) on Modal behind vLLM's OpenAI-compatible API.

    modal deploy deploy/modal_qwen.py          # persistent endpoint (scales to zero when idle)
    modal serve  deploy/modal_qwen.py          # ephemeral dev endpoint (hot-reload)

The endpoint URL is printed on deploy (…modal.run); the OpenAI base is that URL + ``/v1``. Point the
harness at it in ``.env``:

    ENDINGS_PROVIDER=litellm
    ENDINGS_INTERACTOR_MODEL=hosted_vllm/Qwen/Qwen3.5-9B
    ENDINGS_INTERACTOR_API_BASE=https://<workspace>--endings-qwen35-9b-serve.modal.run/v1
    ENDINGS_INTERACTOR_API_KEY=<same value as the Modal secret below>

Auth: the server requires a bearer key read from the Modal Secret ``endings-vllm-api-key``
(key ``VLLM_API_KEY``); create it with ``modal secret create endings-vllm-api-key VLLM_API_KEY=...``.

Serving choices (see docs/decisions/deploy.md):
  * Model weights pinned to an exact Hugging Face revision for reproducibility.
  * vLLM pinned to an exact release (Qwen3.5 support landed after the model's March 2026 release).
  * ``--language-model-only``: the interactor is text-only; skipping the vision tower saves memory.
  * No ``--reasoning-parser``: the ``<think>…</think>`` trace stays inline in ``content`` so the
    harness's ``providers.split_reasoning`` captures it for qualitative analysis (slides, Q2)
    regardless of client library.
  * ``--max-model-len 32768`` rather than the model card's 128K+ recommendation: sessions are short
    (≤ ~40 turns, history window 12) and this fits comfortably on one L40S.
"""

from __future__ import annotations

import os
import subprocess

import modal

MODEL_NAME = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # HF commit, 2026-03-02
VLLM_VERSION = "0.31.0"
GPU = os.environ.get("ENDINGS_MODAL_GPU", "L40S")
MAX_MODEL_LEN = int(os.environ.get("ENDINGS_MODAL_MAX_MODEL_LEN", "32768"))
PORT = 8000
MINUTES = 60
# Idle minutes before the container scales to zero. A cold start costs about as much GPU time
# (4 to 9 minutes), so keeping it warm longer than that only wastes money between runs.
SCALEDOWN_MIN = int(os.environ.get("ENDINGS_MODAL_SCALEDOWN_MIN", "5"))

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("endings-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("endings-vllm-cache", create_if_missing=True)

app = modal.App("endings-qwen35-9b")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MIN * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("endings-vllm-api-key")],
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", str(MAX_MODEL_LEN),
        "--language-model-only",
        "--tensor-parallel-size", "1",
        "--uvicorn-log-level", "info",
    ]
    subprocess.Popen(cmd)
