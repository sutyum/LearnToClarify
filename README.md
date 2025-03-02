# LearnToClarify
RL on verifiable rewards for LLM to learn to ask clarifying questions

```sh
# Standalone Usage
huggingface-cli login
python train.py --use_gsm8k --use_calculator --model_repo_name Technoculture/clarify_0
```

Dataset: [ambiguous_gsm8k](https://huggingface.co/datasets/Technoculture/ambiguous_gsm8k)

## Manual Process
Acquire at least 2 A100 (80GB) GPUs (Can wary depending on the model size)
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/willccbb/verifiers.git
cd verifiers
/root/.local/bin/uv sync
/root/.local/bin/uv pip install flash-attn --no-build-isolation

cd ..
git clone https://github.com/sutyum/LearnToClarify.git
cd LearnToClarify
/root/.local/bin/uv sync

source .venv/bin/activate
accelerate launch --config-file configs/zero3.yaml --num-processes [N-1] verifiers/examples/gsm8k_calculator.py
```

> Tested on Runpod with `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image: Works
