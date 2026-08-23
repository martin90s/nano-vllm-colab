#!/usr/bin/env python3
"""Bootstrap and smoke-test nano-vLLM on a fresh Google Colab GPU VM."""

from __future__ import annotations

import os
import subprocess
import sys


MODEL_ID = "Qwen/Qwen3-0.6B"


def run(*args: str, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True, env=env)


def require_supported_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU is attached. Start this with `colab run --gpu L4 ...`."
        )

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {name} (compute capability {capability[0]}.{capability[1]})")
    if capability[0] < 8:
        raise SystemExit(
            "nano-vLLM's FlashAttention-2 path needs an Ampere-or-newer NVIDIA "
            f"GPU, but Colab assigned {name}. T4 and P100 runtimes are not "
            "compatible; retry when an L4 or A100 is available."
        )


def install_nano_vllm() -> None:
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "packaging",
        "ninja",
        "wheel",
        "transformers==4.51.3",
        "xxhash",
    )

    build_env = os.environ.copy()
    build_env.setdefault("MAX_JOBS", "2")
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-build-isolation",
        "flash-attn",
        env=build_env,
    )
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-deps",
        "git+https://github.com/GeeeekExplorer/nano-vllm.git",
    )


def smoke_test() -> None:
    from huggingface_hub import snapshot_download
    from nanovllm import LLM, SamplingParams

    model_path = snapshot_download(MODEL_ID)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    outputs = llm.generate(
        ["Hello from nano-vLLM on Google Colab. In one sentence, explain CUDA."],
        SamplingParams(temperature=0.6, max_tokens=64),
    )
    print("\nGenerated text:\n", outputs[0]["text"], sep="")


if __name__ == "__main__":
    require_supported_gpu()
    install_nano_vllm()
    smoke_test()
