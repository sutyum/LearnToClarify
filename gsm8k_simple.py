import verifiers as vf

model_name = "chinmaydk99/Qwen2.5-0.5b-GRPO-math"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.MathEnv(dataset="Technoculture/ambiguous_gsm8k", max_steps=5)
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

run_name = "ambiguous_gsm8k_" + model_name.split("/")[-1].lower()

training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=1)

# training_args.max_prompt_length = 128
# training_args.vllm_max_model_len = 1024  # rather than 2048
# training_args.per_device_eval_batch_size = 2  # rather than 8
# training_args.per_device_train_batch_size = 2  # rather than 8
# training_args.num_generations = 2
# training_args.vllm_gpu_memory_utilization = 0.25
# training_args.gradient_accumulation_steps = 1
# training_args.use_liger_kernel = True
# training_args.gradient_checkpointing = True


trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
