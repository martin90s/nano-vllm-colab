# Running nano-vLLM with the Google Colab CLI

This repository requires CUDA, so it cannot run on an Apple Silicon GPU. This
checkout adds a slower PyTorch SDPA attention fallback for Colab's free T4 GPU;
L4 and A100 sessions can still use optional FlashAttention-2 when available.
Free-tier accelerator availability is dynamic and is not guaranteed.

## Free-tier T4 smoke test

Install and authorize the official CLI once:

```bash
uv tool install --force google-colab-cli \
  --with jupyter-kernel-client==0.9.0
colab --auth oauth2 sessions
```

The explicit kernel-client pin works around an API mismatch in Colab CLI 0.6.0
with `jupyter-kernel-client` 1.0.1. Remove the pin once the CLI fixes that
dependency constraint upstream.

Then run the orchestrator from the repository root:

```bash
sh colab/run_free_t4.sh
```

It provisions a T4, uploads the SDPA fallback, installs nano-vLLM, runs a small
Qwen3-0.6B generation, and stops its session on success or failure. T4 is slower
than the upstream FlashAttention-2 path but does not require a paid L4/A100.

## L4/A100 smoke test

If the account has an L4 entitlement, use the upstream optimized path:

```bash
colab run --gpu L4 --timeout 2400 colab/run_nano_vllm.py
```

`colab run` releases its VM after the script exits.

## Keep a session for iterative work

```bash
colab run --gpu L4 --timeout 2400 --keep -s nano-vllm colab/run_nano_vllm.py
colab exec -s nano-vllm --timeout 300 <<'PY'
import torch
print(torch.cuda.get_device_name(0))
PY
colab stop -s nano-vllm
```

Run `colab sessions` at any time to check that no session was left active.

## Codex MCP bridge

Register Google's Colab MCP server once:

```bash
codex mcp add colab-mcp -- \
  uvx --from git+https://github.com/googlecolab/colab-mcp colab-mcp
```

Restart Codex after registration, then ask it to call
`open_colab_browser_connection`. That tool opens a Colab scratch notebook with
a one-time local bridge token. Once the page connects, the MCP server
dynamically adds its notebook editing and execution tools.
