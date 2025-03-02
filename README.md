# LearnToClarify
RL on verifiable rewards for LLM to learn to ask clarifying questions

```sh
huggingface-cli login
python train.py --use_gsm8k --use_calculator --model_repo_name Technoculture/clarify_0
```

Dataset: [ambiguous_gsm8k](https://huggingface.co/datasets/Technoculture/ambiguous_gsm8k)
