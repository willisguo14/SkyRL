set -x


# export WANDB_API_KEY=<your_key_here>


: "${USER_DIR:="/data/user_data/willisg"}"
: "${DATA_DIR:="/data/user_data/willisg/cruxeval"}"
: "${NUM_GPUS:=2}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"
# : "${INFERENCE_BACKEND:=sglang}"

uv run --isolated --extra $INFERENCE_BACKEND -m examples.code_understanding.crux_eval.main_cruxeval \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet', '/data/user_data/willisg/gsm8k/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="/data/user_data/willisg/ckpts_hf/gsm8k/global_step_80/policy/" \
  trainer.placement.colocate_all=true \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=200 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=20 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=384 \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=cruxeval \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="cruxeval" \
  trainer.run_name="cruxeval_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$USER_DIR/ckpts/cruxeval" \
  $@
