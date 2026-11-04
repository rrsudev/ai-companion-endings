"""Serve the judge model (google/gemma-4-26B-A4B-it) on Modal behind vLLM's OpenAI-compatible API.

    modal deploy deploy/modal_gemma.py          # persistent endpoint (scales to zero when idle)
    modal serve  deploy/modal_gemma.py          # ephemeral dev endpoint (hot-reload)

Same layout as ``modal_qwen.py``. Differences:
  * The weights are Apache-2.0 and not gated, so no Hugging Face token is needed.
  * 26B parameters in bf16 are about 52 GB of weights, which does not fit the L40S (48 GB) used for
    the other roles. The default GPU is an A100-80GB; override with ``ENDINGS_MODAL_GPU``.
  * Endpoint auth reuses the Modal Secret ``endings-vllm-api-key``.
  * Weights pinned to the HF commit below for reproducibility.

Point the harness at it in ``.env``:

    ENDINGS_JUDGE_MODEL=hosted_vllm/google/gemma-4-26B-A4B-it
    ENDINGS_JUDGE_API_BASE=https://<workspace>--endings-gemma4-26b-serve.<region>.modal.run/v1
    ENDINGS_JUDGE_API_KEY=<same value as the Modal secret>
"""

from __future__ import annotations

import os
import subprocess

import modal

MODEL_NAME = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"  # HF commit, lastModified 2026-07-20
VLLM_VERSION = "0.31.0"
GPU = os.environ.get("ENDINGS_MODAL_GPU", "A100-80GB")
MAX_MODEL_LEN = int(os.environ.get("ENDINGS_MODAL_MAX_MODEL_LEN", "32768"))
PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("endings-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("endings-vllm-cache", create_if_missing=True)

app = modal.App("endings-gemma4-26b")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=10 * MINUTES,  # idle time before scaling to zero (cost control)
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
