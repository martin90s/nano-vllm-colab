#!/bin/sh
set -eu

session="nano-vllm-t4"

cleanup() {
  colab stop -s "$session" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

colab new --gpu T4 -s "$session"
colab exec -s "$session" --timeout 120 <<'PY'
import pathlib
import subprocess

repo = pathlib.Path("/content/nano-vllm")
if not repo.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/GeeeekExplorer/nano-vllm.git",
            str(repo),
        ],
        check=True,
    )
subprocess.run(
    ["git", "checkout", "bb823b3e06983d71485a8e1f23715ebd87d98ef8"],
    cwd=repo,
    check=True,
)
PY
colab upload -s "$session" \
  nanovllm/layers/attention.py \
  /content/nano-vllm/nanovllm/layers/attention.py
colab upload -s "$session" \
  nanovllm/engine/model_runner.py \
  /content/nano-vllm/nanovllm/engine/model_runner.py
colab upload -s "$session" pyproject.toml /content/nano-vllm/pyproject.toml
colab exec -s "$session" --file colab/remote_t4_smoke.py --timeout 2400
