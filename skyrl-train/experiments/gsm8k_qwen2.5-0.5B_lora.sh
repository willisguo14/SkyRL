export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_GPUS=4
export LOGGER=wandb
export INFERENCE_BACKEND=vllm

group="qwen0.5b_lora_rank_sweep"
lr="1e-5"
for r in 1; do
  name="qwen0.5b_lora_r${r}_lr${lr}"
  WANDB_PROJECT="gsm8k_lora" WANDB_GROUP="$group" WANDB_NAME="$name" \
  bash /work5/nkale/ml_projects/SkyRL/skyrl-train/examples/gsm8k/run_gsm8k.sh \
    trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    trainer.policy.model.lora.rank=$r \
    trainer.policy.optimizer_config.lr=$lr \
    trainer.run_name="$name" \
    trainer.project_name="gsm8k_lora"
done