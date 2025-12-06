"""
Training entrypoint for code understanding task.

Usage:
    uv run --isolated --extra vllm -m examples.code_understanding.main_code_understanding
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the code_understanding environment inside the entrypoint task
    register(
        id="code_understanding",
        entry_point="examples.code_understanding.env:CodeUnderstandingEnv",
    )

    # Run the training loop
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
