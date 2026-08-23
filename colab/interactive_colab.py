#!/usr/bin/env python3
"""Load nano-vLLM once and display an interactive prompt UI in Colab."""

from __future__ import annotations

import torch
from IPython.display import clear_output, display
import ipywidgets as widgets
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


MODEL_ID = "Qwen/Qwen3-0.6B"


if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU is attached. Choose Runtime > Change runtime type > T4 GPU, "
        "then run the notebook again."
    )

capability = torch.cuda.get_device_capability(0)
if capability < (7, 5):
    raise RuntimeError("The interactive demo requires a T4 or newer GPU.")

try:
    from google.colab import output as colab_output

    colab_output.enable_custom_widget_manager()
except ImportError:
    pass

print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0)}...")
model_path = snapshot_download(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
    model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=16,
    gpu_memory_utilization=0.8,
)
print("Model loaded. Enter a prompt below.")

prompt_box = widgets.Textarea(
    value="Explain CUDA in one short paragraph.",
    description="Prompt:",
    layout=widgets.Layout(width="100%", height="120px"),
    style={"description_width": "70px"},
)
temperature_slider = widgets.FloatSlider(
    value=0.6,
    min=0.1,
    max=1.5,
    step=0.1,
    description="Temperature:",
    continuous_update=False,
    style={"description_width": "90px"},
)
max_tokens_slider = widgets.IntSlider(
    value=128,
    min=16,
    max=256,
    step=16,
    description="Max tokens:",
    continuous_update=False,
    style={"description_width": "90px"},
)
generate_button = widgets.Button(
    description="Generate",
    button_style="primary",
    icon="play",
)
result_panel = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #d0d7de",
        padding="12px",
        width="100%",
    )
)


def generate(_button: widgets.Button) -> None:
    prompt = prompt_box.value.strip()
    if not prompt:
        with result_panel:
            clear_output(wait=True)
            print("Enter a prompt first.")
        return

    generate_button.disabled = True
    with result_panel:
        clear_output(wait=True)
        print("Generating...")

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate(
            [formatted_prompt],
            SamplingParams(
                temperature=float(temperature_slider.value),
                max_tokens=int(max_tokens_slider.value),
            ),
        )
        with result_panel:
            clear_output(wait=True)
            print(outputs[0]["text"].strip())
    except Exception as error:
        with result_panel:
            clear_output(wait=True)
            print(f"Generation failed: {error}")
    finally:
        generate_button.disabled = False


generate_button.on_click(generate)
display(
    widgets.VBox(
        [
            prompt_box,
            widgets.HBox([temperature_slider, max_tokens_slider]),
            generate_button,
            result_panel,
        ],
        layout=widgets.Layout(width="100%"),
    )
)
