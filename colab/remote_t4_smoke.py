#!/usr/bin/env python3
"""Install this checkout and smoke-test the SDPA fallback on a Colab T4."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_PATH = Path("/content/nano-vllm")
MODEL_ID = "Qwen/Qwen3-0.6B"


def run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    result = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, flush=True)
    result.check_returncode()


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is attached to this Colab session.")
    capability = torch.cuda.get_device_capability(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"(compute capability {capability[0]}.{capability[1]})"
    )
    if capability < (7, 5):
        raise SystemExit("The free-tier fallback requires a T4 or newer GPU.")
    if not REPO_PATH.is_dir():
        raise SystemExit(f"Expected the uploaded checkout at {REPO_PATH}.")

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "transformers==4.51.3",
        "xxhash",
    )
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--editable",
        str(REPO_PATH),
    )

    sys.path.insert(0, str(REPO_PATH))

    from huggingface_hub import snapshot_download
    from nanovllm import LLM, SamplingParams

    model_path = snapshot_download(MODEL_ID)
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=16,
        gpu_memory_utilization=0.8,
    )
    outputs = llm.generate(
        ["Hello from nano-vLLM on a free Colab T4. In one sentence, explain CUDA."],
        SamplingParams(temperature=0.6, max_tokens=64),
    )
    print("\nGenerated text:\n", outputs[0]["text"], sep="")


if __name__ == "__main__":
    main()
