export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NUM_GPUS=4
export LOGGER=wandb
export INFERENCE_BACKEND=vllm
export HOME='/work5/nkale/ml_projects/SkyRL'

group="qwen0.5b_lora_rank_sweep"
lr="1e-5"
allowed_max_ranks=(8 16 32 64 128 256 320 512)

for r in 2 4 8; do
  name="qwen0.5b_lora_r${r}_lr${lr}"
  WANDB_PROJECT="gsm8k_lora" WANDB_GROUP="$group" WANDB_NAME="$name" \
  NUM_GPUS=$NUM_GPUS \
  bash /work5/nkale/ml_projects/SkyRL/skyrl-train/examples/gsm8k/run_gsm8k.sh \
    trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    trainer.policy.model.lora.rank=$r \
    trainer.policy.optimizer_config.lr=$lr \
    trainer.run_name="$name" \
    trainer.project_name="gsm8k_lora" \
    +generator.engine_init_kwargs.max_lora_rank=8
done