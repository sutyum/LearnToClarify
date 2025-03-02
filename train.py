"""
Offer the following training variants:
- First on GSM8k dataset then on ambiguous-gsm8k dataset or directly on ambiguous-gsm8k dataset
- With or without tools
"""

import os
import argparse
import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import SEARCH_FEW_SHOT
import logging
from huggingface_hub import HfApi, create_repo

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Train model using Verifier-based Reinforcement Learning"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-1.5B-Instruct",
    # default="Qwen/Qwen2.5-7B-Instruct",
    help="Model name to train. Default: Qwen/Qwen2.5-7B-Instruct",
)
parser.add_argument(
    "--max_steps", type=int, default=3, help="Number of steps to train. Default: 3"
)
parser.add_argument(
    "--num_gpus", type=int, default=2, help="Number of GPUs to use. Default: 2"
)
parser.add_argument(
    "--use_gsm8k",
    action="store_true",
    help="Train sequentially on GSM8k then ambiguous-gsm8k",
)
parser.add_argument(
    "--use_calculator", action="store_true", help="Use calculator tool during training"
)
parser.add_argument(
    "--save_checkpoints",
    action="store_true",
    help="Save model checkpoints after each dataset",
)
parser.add_argument(
    "--model_repo_name",
    type=str,
    default="Technoculture/trained_model",
    help="Hugging Face repository name to upload the trained model (e.g., 'username/repo')",
)

args = parser.parse_args()
logging.info(f"Arguments: {args}")

# Load model and tokenizer
model_name = args.model_name
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Set tools based on user input
tools = [calculator] if args.use_calculator else []

# Determine datasets for training
datasets = ["gsm8k", "ambiguous_gsm8k"] if args.use_gsm8k else ["ambiguous_gsm8k"]

# Train sequentially on each dataset
for dataset in datasets:
    try:
        logging.info(f"Starting training on {dataset} with tools: {bool(tools)}")

        # Initialize environment with the current dataset
        vf_env = vf.ToolEnv(dataset=dataset, tools=tools, max_steps=args.max_steps)

        # Dynamic run name based on dataset and tool usage
        run_name = f"{dataset}_training{'_with_tools' if tools else ''}"

        # Configure and initialize trainer
        trainer = vf.GRPOEnvTrainer(
            model=model,
            processing_class=tokenizer,
            env=vf_env,
            reward_funcs=vf_env.get_rubric(),
            args=vf.get_default_grpo_config(run_name=run_name, num_gpus=args.num_gpus),
            train_dataset=vf_env.get_dataset(),
        )

        # Train the model
        trainer.train()

        # Save checkpoint if requested
        if args.save_checkpoints:
            checkpoint_path = f"{dataset}_checkpoint"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")

    except Exception as e:
        logging.error(f"Error during training on {dataset}: {e}")
        raise

# After training, save and upload the final model to Hugging Face
try:
    # Save the final model locally
    final_model_path = "final_trained_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Final model saved locally at {final_model_path}")

    # Create the repository if it doesn’t exist
    create_repo(args.model_repo_name, exist_ok=True)

    # Upload the model to Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=final_model_path, repo_id=args.model_repo_name, repo_type="model", token=os.environ["HF_TOKEN"]
    )
    logging.info(f"Model uploaded to: https://huggingface.co/{args.model_repo_name}")

except Exception as e:
    logging.error(f"Error uploading model to Hugging Face: {e}")

logging.info("Training and upload completed successfully.")
