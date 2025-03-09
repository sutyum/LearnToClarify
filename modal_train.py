import modal

# Create a clean image with explicit installation steps
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .apt_install("git")
    .pip_install("uv")
    # First remove any existing installations
    .run_commands("rm -rf /root/verifiers || true")
    .run_commands("pip uninstall -y verifiers || true")
    # Clone and install the repository correctly
    .run_commands("cd /root && git clone https://github.com/sutyum/verifiers.git", force_build=True)
    # Install the package in development mode with pip (not uv)
    .run_commands("cd /root/verifiers && pip install -e .")
    # Install other dependencies
    .run_commands("pip install torch accelerate flash-attn")
    .add_local_file("zero3.yaml", "/root/LearnToClarify/zero3.yaml")
    .add_local_file("gsm8k_simple.py", "/root/LearnToClarify/gsm8k_simple.py")
)

app = modal.App("verifiers-training")

PROC_COUNT = 2

@app.function(
    image=image,
    gpu=f"A100:{PROC_COUNT}",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=4 * 60 * 60,
)
def train():
    # import os
    import sys
    import subprocess
    
    # Debug: Print Python package information
    print("Python executable:", sys.executable)
    print("Python path:", sys.path)
    
    # Test verifiers import directly
    print("\n=== Testing verifiers import ===")
    try:
        import verifiers
        print("Verifiers package location:", verifiers.__file__)
        from verifiers.envs.environment import Environment
        print("Successfully imported Environment:", Environment)
    except Exception as e:
        print(f"Import error: {e}")
    
    # Copy your gsm8k_simple.py to the root
    print("\n=== Creating runner script ===")
    with open("/root/gsm8k_simple.py", "w") as f:
        with open("/root/LearnToClarify/gsm8k_simple.py", "r") as src:
            f.write(src.read())

    # Try running directly first
    try:
        print("\n=== Running script directly ===")
        subprocess.run("cd /root/LearnToClarify && python gsm8k_simple.py", shell=True, check=False)
    except Exception as e:
        print(f"Direct execution failed: {e}")

    # Now try with accelerate
    print("\n=== Running with accelerate ===")
    cmd = (
        "cd /root && "
        "PYTHONPATH=/root:/root/verifiers "
        f"accelerate launch --config-file /root/LearnToClarify/zero3.yaml --num-processes {PROC_COUNT} /root/LearnToClarify/gsm8k_simple.py"
    )
    subprocess.run(cmd, shell=True, check=True)
    
    return "Training completed"

if __name__ == "__main__":
    with app.run():
        train.remote()
