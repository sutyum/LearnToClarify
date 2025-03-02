import modal

# Define the custom image
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .apt_install("git")
    .pip_install("uv")
    .pip_install("accelerate")

    .workdir("/root")
    .add_local_file("pyproject.toml", "/root/LearnToClarify/pyproject.toml", copy=True)
    .add_local_file("zero3.yaml", "/root/LearnToClarify/zero3.yaml", copy=True)
    .add_local_file("train.py", "/root/LearnToClarify/train.py", copy=True)
    .add_local_file("gsm8k_simple.py", "/root/LearnToClarify/gsm8k_simple.py", copy=True)

    .run_commands("cd /root/LearnToClarify && uv sync", force_build=True)
    # .run_commands("uv pip install flash-attn --no-build-isolation")
)

# Define the app
app = modal.App("verifiers-training")

# Define the training function
@app.function(
    image=image,
    gpu="A100:1",  # Use 2 A100 GPUs
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    timeout=4 * 60 * 60,  # 4 hours
)
def train():
    import subprocess

    # Command to activate virtual environment and run training
    cmd = (
        "cd /root/LearnToClarify && uv run accelerate launch --config-file zero3.yaml --num-processes 1 gsm8k_simple.py"
        # "cd /root/LearnToClarify && uv run accelerate launch --config-file zero3.yaml --num-processes 2 train.py"
    )
    subprocess.run(cmd, shell=True, check=True)

# Optional: Local entrypoint for testing
if __name__ == "__main__":
    with app.run():
        train.local()
