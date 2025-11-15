PiSSA Training in SkyRL
=======================

This guide demonstrates how to use PiSSA (Principal Singular Values and Singular Vectors Adaptation) for efficient reinforcement learning training in SkyRL. PiSSA is an advanced variant of LoRA that provides better initialization through SVD decomposition, potentially leading to faster convergence and better performance.

What is PiSSA?
--------------

PiSSA (Principal Singular Values and Singular Vectors Adaptation) is a parameter-efficient fine-tuning technique that improves upon LoRA's random initialization by using Singular Value Decomposition (SVD).

**Key Differences from LoRA:**

- **LoRA**: Initializes adapters randomly (B=0, A~N(0,σ)) while keeping base weights W frozen
- **PiSSA**: Uses SVD to split W = W_residual + W_principal, where:

  - W_residual becomes the new frozen base weights
  - Adapters are initialized to W_principal (the principal singular components)
  - This provides a better starting point for fine-tuning

**Benefits:**

1. **Better initialization**: Adapters start from principal components rather than random values
2. **Faster convergence**: Better initialization can lead to faster training
3. **Compatible with LoRA**: After initialization, training is identical to LoRA
4. **Drop-in replacement**: Can be used anywhere LoRA is used with minimal config changes

Configuration
-------------

PiSSA can be configured for both policy and critic models in the training configuration. Here's how to set it up:

.. code-block:: yaml

    trainer:
      policy:
        model:
          path: "Qwen/Qwen2.5-1.5B-Instruct"
          lora:
            rank: 32              # LoRA rank (higher = more parameters)
            alpha: 32             # LoRA scaling parameter
            dropout: 0            # LoRA dropout rate
            init_method: "pissa"  # PiSSA initialization method
            lora_sync_path: "/tmp/skyrl_lora_sync"  # Path for adapter sync
      critic:
        model:
          path: "Qwen/Qwen2.5-1.5B-Instruct"
          lora:
            rank: 32
            alpha: 32
            dropout: 0
            init_method: "pissa"

Key Parameters
~~~~~~~~~~~~~~

- **``rank``**: The rank of the low-rank decomposition. Higher values mean more trainable parameters but also more memory usage. Common values are 8, 16, 32, or 64.
- **``alpha``**: The scaling parameter for LoRA. Often set equal to rank, but can be tuned independently.
- **``dropout``**: Dropout rate applied to LoRA layers. Helps with regularization.
- **``init_method``**: The initialization method for adapters. Options:

  - ``"default"``: Standard LoRA random initialization
  - ``"pissa"``: PiSSA SVD-based initialization

- **``lora_sync_path``**: Directory path where LoRA adapters are saved and synchronized between training and inference engines.

Target Modules
~~~~~~~~~~~~~~

By default, PiSSA is applied to all linear layers in the model. You can customize which modules to target:

.. code-block:: yaml

    trainer:
      target_modules: "all-linear"  # Apply to all linear layers OR
      # specify specific modules as a list
      exclude_modules: null  # Modules to exclude from LoRA/PiSSA

Running PiSSA Training
----------------------

Here's a complete example of running PiSSA training on GSM8K:

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

First, prepare your dataset:

.. code-block:: bash

   uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

Training Script
~~~~~~~~~~~~~~~

Create a training script with PiSSA configuration:

.. code-block:: bash

   #!/bin/bash
   set -x

   DATA_DIR="$HOME/data/gsm8k"
   NUM_GPUS=4
   LOGGER="wandb"  # change to "console" to print to stdout
   INFERENCE_BACKEND="vllm"

   CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
     data.train_data="['$DATA_DIR/train.parquet']" \
     data.val_data="['$DATA_DIR/validation.parquet']" \
     trainer.algorithm.advantage_estimator="grpo" \
     trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
     trainer.policy.model.lora.rank=32 \
     trainer.policy.model.lora.alpha=32 \
     trainer.policy.model.lora.init_method="pissa" \
     trainer.policy.model.lora.lora_sync_path="/tmp/skyrl_lora_sync" \
     trainer.strategy=fsdp2 \
     trainer.placement.colocate_all=true \
     trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
     trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
     generator.num_inference_engines=$NUM_GPUS \
     generator.inference_engine_tensor_parallel_size=1 \
     trainer.train_batch_size=128 \
     trainer.policy_mini_batch_size=128 \
     trainer.micro_forward_batch_size_per_gpu=64 \
     trainer.micro_train_batch_size_per_gpu=64 \
     trainer.ckpt_interval=10 \
     generator.sampling_params.max_generate_length=1024 \
     trainer.policy.optimizer_config.lr=3.0e-5 \
     trainer.algorithm.use_kl_loss=true \
     generator.backend=$INFERENCE_BACKEND \
     generator.batched=true \
     environment.env_class=gsm8k \
     generator.n_samples_per_prompt=4 \
     trainer.logger="$LOGGER" \
     trainer.project_name="gsm8k_0.5b_pissa" \
     trainer.run_name="gsm8k_0.5b_pissa_test" \
     trainer.ckpt_path="$HOME/ckpts/gsm8k_0.5b_pissa_ckpt"

Launch Training
~~~~~~~~~~~~~~~

Set up your WandB API key and run the training:

.. code-block:: bash

   export WANDB_API_KEY=your_wandb_api_key
   bash examples/pissa/run_qwen2_5_0.5b_gsm8k_grpo_pissa.sh

How PiSSA Weight Sync Works
----------------------------

PiSSA requires special handling during the first weight synchronization:

1. **Initialization**: PEFT library performs SVD decomposition to split W → W_residual + adapters
2. **First sync**: Both W_residual (modified base weights) and adapters are synced to vLLM
3. **Subsequent syncs**: Only adapters are synced (same as LoRA)

This two-step sync ensures vLLM has the correct base weights that were modified during PiSSA initialization.

**Implementation details:**

- Base weights are extracted from the PEFT model, cleaned of internal PEFT naming (``.base_layer.``), and synced
- After base sync completes, adapters are saved to disk and loaded by vLLM

Configuration Tips
------------------

1. **Learning rate:** Use similar learning rates as LoRA (roughly 10× higher than full fine-tuning)

2. **Rank:** Choose a rank large enough to capture the principal components. Start with 32-64 for most RL fine-tuning tasks.

3. **Layer coverage:** Apply PiSSA to *all* layers, particularly MLP/MoE layers — attention-only PiSSA tends to underperform.

4. **Initialization time:** PiSSA initialization with SVD takes 2-5 minutes for large models. This is a one-time cost at startup.

Checkpointing
-------------

PiSSA checkpoints work the same as LoRA checkpoints. The PEFT configuration (including the PiSSA initialization method) is automatically saved with the model weights and restored when you resume training.

Current Limitations
-------------------

SkyRL's PiSSA implementation has the following current limitations:

1. **Disk-based synchronization**: Like LoRA, adapters are saved to disk and reloaded rather than synchronized in-memory.

2. **First sync overhead**: The first weight sync includes both base weights and adapters, which takes slightly longer than subsequent adapter-only syncs.

3. **Single adapter per model**: Currently, only one PiSSA adapter can be active per model at a time.

These limitations are being addressed in future releases, with plans for in-memory synchronization and improved adapter management.

References
----------

- PiSSA paper: `Junjie Oscar Yin et al., "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models" <https://arxiv.org/abs/2404.02948>`_
- PEFT library documentation: `Hugging Face PEFT <https://huggingface.co/docs/peft>`_
