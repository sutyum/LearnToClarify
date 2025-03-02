import modal

config_content = """compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false"""

# Define the custom image
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    # Clone both repositories
    .run_commands("git clone https://github.com/willccbb/verifiers.git /verifiers")
    .run_commands("git clone https://github.com/sutyum/LearnToClarify.git /LearnToClarify")
    # Install dependencies for verifiers
    .run_commands("cd /verifiers && .venv/bin/uv sync")
    .run_commands("cd /LearnToClarify && .venv/bin/uv sync")
    # Install flash-attn into LearnToClarify's virtual environment
    .run_commands("cd /LearnToClarify && .venv/bin/uv pip install flash-attn --no-build-isolation")
)

# Define the app
app = modal.App("verifiers-training")

# Define the training function
@app.function(
    image=image,
    gpu="A100:2",  # Use 2 A100 GPUs
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],  # For HF_TOKEN
    timeout=4 * 60 * 60,  # 4 hours
)
def train():
    import subprocess
    # Command to activate virtual environment and run training
    cmd = (
        "source /LearnToClarify/.venv/bin/activate && "
        "accelerate launch --config-file /LearnToClarify/zero3.yaml --num-processes 2 /LearnToClarify/train.py"
    )
    subprocess.run(cmd, shell=True, check=True)

# Optional: Local entrypoint for testing
if __name__ == "__main__":
    with app.run():
        train.local()
