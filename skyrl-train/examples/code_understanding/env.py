from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
import ast

def extract_solution(solution_str):
    return solution_str.split("####")[-1].strip()

def compute_score(solution_str, ground_truth):
    answer = extract_solution(solution_str)
    
    try:
        if ast.literal_eval(answer) == ast.literal_eval(ground_truth):
            return 1.0
        else:
            return 0.0
    except:
        return 0.0

class CodeUnderstandingEnv(BaseTextEnv):
    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
    
    def _get_reward(self, action: str) -> float:
        return compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={}
        )