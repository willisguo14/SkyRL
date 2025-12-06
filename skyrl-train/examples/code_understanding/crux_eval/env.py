from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
from .utils import compute_score


class CruxEvalEnv(BaseTextEnv):
    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        assert "extra_info" in extras, "extra_info field is required"
        assert "task_type" in extras["extra_info"], "task_type is required in extra_info field"
        assert "code" in extras["extra_info"], "code is required in extra_info field"

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.task_type = extras["extra_info"]["task_type"]
        self.code = extras["extra_info"]["code"]

    def _get_reward(self, action: str) -> float:
        return compute_score(
            response=action,
            code=self.code,
            ground_truth=self.ground_truth,
            task_type=self.task_type,
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True 
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={"task_type": self.task_type}
        )
