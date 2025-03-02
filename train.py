"""
Offer the following training variants:
- First on GSM8k dataset then on ambiguous-gsm8k dataset or directly on ambiguous-gsm8k dataset
- With or without tools
"""
import argparse
import verifiers as vf
from verifiers.tools import calculator
# from verifiers.prompts import SEARCH_FEW_SHOT


parser = argparse.ArgumentParser(
    description="Train model using Verifier based Reinforcement Learning"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Model name to train. Default: Qwen/Qwen2.5-7B-Instruct",
)
parser.add_argument(
    "--max_steps", type=int, default=3, help="Number of steps to train. Default: 3"
)
parser.add_argument(
    "--num_gpus", type=int, default=2, help="Number of GPUs to use. Default: 2"
)
parser.add_argument(
    "--use_gsm8k", action="store_true", help="Use GSM8k dataset"
)
parser.add_argument(
    "--use_calculator", action="store_true", help="Use calculator tool"
)

args = parser.parse_args()

model_name = args.model_name
model, tokenizer = vf.get_model_and_tokenizer(model_name)

tools = []
if args.use_calculator:
    tools = [calculator]

datasets = []
if args.use_gsm8k:
    datasets = ["gsm8k", "ambiguous_gsm8k"]
else:
    datasets = ["ambiguous_gsm8k"]

for dataset in datasets:
    vf_env = vf.ToolEnv(
        dataset="gsm8k", 
        # few_shot=SEARCH_FEW_SHOT[0], 
        tools=tools,
        max_steps=args.max_steps
    )

    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        reward_funcs=vf_env.get_rubric(),
        args=vf.get_default_grpo_config(run_name="ambiguous_gsm8k", num_gpus=args.num_gpus),
        train_dataset=vf_env.get_dataset(),
    )

    trainer.train()
