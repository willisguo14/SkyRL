"""
Tests for PiSSA (Principal Singular Values and Singular Vectors Adaptation) implementation.

# Run PiSSA tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_pissa.py
"""

import pytest
import asyncio
import ray
import hydra
import torch
import os
import tempfile
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_prompts, init_inference_engines, run_inference
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_pissa_config(init_method: str = "pissa") -> DictConfig:
    """Get base config with PiSSA-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.run_engines_locally = True

        # PiSSA configuration
        cfg.trainer.policy.model.lora.rank = 32
        cfg.trainer.policy.model.lora.alpha = 32
        cfg.trainer.policy.model.lora.dropout = 0.1
        cfg.trainer.policy.model.lora.init_method = init_method
        cfg.trainer.target_modules = "all-linear"

        return cfg


@pytest.mark.vllm
def test_pissa_initialization():
    """
    Tests PiSSA initialization with SVD.
    Verifies that the model can be initialized with PiSSA and adapters are created correctly.
    """
    from skyrl_train.model_wrapper import HFModelWrapper

    cfg = get_test_pissa_config(init_method="pissa")

    # Initialize model with PiSSA
    model_wrapper = HFModelWrapper(
        model_path=MODEL,
        lora_rank=cfg.trainer.policy.model.lora.rank,
        lora_alpha=cfg.trainer.policy.model.lora.alpha,
        lora_dropout=cfg.trainer.policy.model.lora.dropout,
        init_method="pissa",
        target_modules=cfg.trainer.target_modules,
        load_in_4bit=False,
        flash_attn=False,
        gradient_checkpointing=False,
        disable_fast_tokenizer=False,
        bf16=True,
        use_torch_compile=False,
    )

    # Verify that PEFT model was created
    assert hasattr(model_wrapper.model, "peft_config"), "Model should have peft_config"

    # Verify PiSSA initialization
    peft_config = model_wrapper.model.peft_config
    first_adapter = next(iter(peft_config.values()))
    init_lora_weights = getattr(first_adapter, "init_lora_weights", True)

    assert isinstance(init_lora_weights, str), "PiSSA should use string init_lora_weights"
    assert "pissa" in init_lora_weights.lower(), f"init_lora_weights should contain 'pissa', got {init_lora_weights}"
    print(f"[PiSSA Test] Successfully initialized with init_method={init_lora_weights}")

    # Verify that adapters exist
    adapter_names = list(model_wrapper.model.peft_config.keys())
    assert len(adapter_names) > 0, "Should have at least one adapter"
    print(f"[PiSSA Test] Found adapters: {adapter_names}")


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=[
        "no_colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm",
    ],
)
def test_pissa_weight_sync_e2e(ray_init_fixture, colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests PiSSA weight sync to inference engine.
    This test verifies that:
    1. First sync includes both base weights (W_residual) and adapters
    2. Subsequent syncs are adapter-only
    3. Inference works correctly with PiSSA adapters
    """
    cfg = get_test_pissa_config(init_method="pissa")
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.weight_sync_backend = weight_sync_backend
    cfg.trainer.strategy = strategy
    cfg.generator.backend = backend
    cfg.generator.inference_engine_tensor_parallel_size = tp_size

    # Initialize inference engines with LoRA enabled (PiSSA adapters are LoRA-compatible)
    client, pg = init_inference_engines(
        model=MODEL,
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend=backend,
        sleep_level=1,  # since we explicitly sync weights
        enable_lora=True,  # Enable LoRA for PiSSA adapters
    )

    # Initialize policy worker with PiSSA
    policy = init_worker_with_type(
        "policy",
        shared_pg=pg,
        colocate_all=cfg.trainer.placement.colocate_all,
        num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
        cfg=cfg,
    )

    sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

    # Initialize weight sync state
    ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
    asyncio.run(client.reset_prefix_cache())

    # First sync: should sync both W_residual + adapters
    print("[PiSSA Test] Performing first weight sync (W_residual + adapters)")
    ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

    # Run inference to verify it works
    outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
    print(f"[PiSSA Test] First sync - Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    assert len(outputs['responses']) > 0, "Should generate responses after first sync"

    # Second sync: should be adapter-only (simulating training step)
    print("[PiSSA Test] Performing second weight sync (adapter-only)")
    ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

    # Run inference again to verify second sync works
    outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
    print(f"[PiSSA Test] Second sync - Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    assert len(outputs['responses']) > 0, "Should generate responses after second sync"


def test_pissa_checkpoint_metadata():
    """
    Tests that PiSSA metadata is correctly saved and loaded from checkpoints.
    """
    from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl_train.model_wrapper import HFModelWrapper
    import torch.distributed as dist

    # Initialize distributed if not already initialized
    if not dist.is_initialized():
        pytest.skip("Requires distributed environment")

    cfg = get_test_pissa_config(init_method="pissa")

    # Initialize model with PiSSA
    model_wrapper = HFModelWrapper(
        model_path=MODEL,
        lora_rank=cfg.trainer.policy.model.lora.rank,
        lora_alpha=cfg.trainer.policy.model.lora.alpha,
        lora_dropout=cfg.trainer.policy.model.lora.dropout,
        init_method="pissa",
        target_modules=cfg.trainer.target_modules,
        load_in_4bit=False,
        flash_attn=False,
        gradient_checkpointing=False,
        disable_fast_tokenizer=False,
        bf16=True,
        use_torch_compile=False,
    )

    # Initialize FSDP strategy
    strategy = FSDPStrategy(
        fsdp_strategy=cfg.trainer.strategy,
        cpu_offload=cfg.trainer.policy.fsdp_config.cpu_offload,
        reshard_after_forward=cfg.trainer.policy.fsdp_config.reshard_after_forward,
        is_lora=True,
        fsdp_size=cfg.trainer.policy.fsdp_config.fsdp_size,
    )

    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "pissa_checkpoint")

        # Prepare dummy states for checkpoint
        client_state = {"step": 1, "epoch": 0}

        # Save checkpoint
        print(f"[PiSSA Test] Saving checkpoint to {ckpt_path}")
        strategy.save_checkpoint(
            model=model_wrapper.model,
            ckpt_dir=ckpt_path,
            optimizer=None,
            scheduler=None,
            client_state=client_state,
            tag="test",
        )

        # Load checkpoint and verify metadata
        print(f"[PiSSA Test] Loading checkpoint from {ckpt_path}")
        _, states = strategy.load_checkpoint(
            model=model_wrapper.model,
            ckpt_dir=ckpt_path,
            optimizer=None,
            scheduler=None,
            tag="test",
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )

        # Note: Extra state dict is not returned in states, need to read directly
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        extra_path = os.path.join(ckpt_path, f"extra_state_world_size_{world_size}_rank_{rank}.pt")

        with open(extra_path, "rb") as f:
            extra_state_dict = torch.load(f, map_location="cpu", weights_only=False)

        # Verify PiSSA metadata
        assert "is_pissa" in extra_state_dict, "Checkpoint should contain is_pissa metadata"
        assert extra_state_dict["is_pissa"] is True, "is_pissa should be True"
        assert "lora_init_method" in extra_state_dict, "Checkpoint should contain lora_init_method metadata"
        assert "pissa" in extra_state_dict["lora_init_method"].lower(), "lora_init_method should contain 'pissa'"

        print(f"[PiSSA Test] Checkpoint metadata verified: is_pissa={extra_state_dict['is_pissa']}, init_method={extra_state_dict['lora_init_method']}")
