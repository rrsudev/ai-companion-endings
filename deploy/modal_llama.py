"""Serve the interactor model (meta-llama/Llama-3.1-8B-Instruct) on Modal behind vLLM's OpenAI API.

    modal deploy deploy/modal_llama.py          # persistent endpoint (scales to zero when idle)
    modal serve  deploy/modal_llama.py          # ephemeral dev endpoint (hot-reload)

Same layout as ``modal_qwen.py``. Differences:
  * The model is gated on Hugging Face. Accept the Llama 3.1 license on the model page, then create
    a Modal Secret holding a read token:
        modal secret create endings-hf-token HF_TOKEN=hf_...
  * Auth for the endpoint itself reuses the existing Modal Secret ``endings-vllm-api-key``.
  * No ``<think>`` trace: Llama 3.1 is not a reasoning model, so replies carry no reasoning trace.
  * ``MODEL_REVISION`` is not pinned yet. Pin it to the commit that ran before reporting results.

Point the harness at it in ``.env``:

    ENDINGS_INTERACTOR_MODEL=hosted_vllm/meta-llama/Llama-3.1-8B-Instruct
    ENDINGS_INTERACTOR_API_BASE=https://<workspace>--endings-llama31-8b-serve.<region>.modal.run/v1
"""

from __future__ import annotations

import os
import subprocess

import modal

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = os.environ.get("ENDINGS_LLAMA_REVISION")  # None = latest; pin before real runs
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

app = modal.App("endings-llama31-8b")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MIN * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[
        modal.Secret.from_name("endings-vllm-api-key"),
        modal.Secret.from_name("endings-hf-token"),
    ],
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", str(MAX_MODEL_LEN),
        "--tensor-parallel-size", "1",
        "--uvicorn-log-level", "info",
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
