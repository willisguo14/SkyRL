"""
Environment for code understanding tasks.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
import re


def extract_solution(solution_str):
    """Extract the final answer after #### using strict matching."""
    # Look for the pattern #### followed by any text until the end
    solution = re.search(r"####\s*(.+?)(?:\n|$)", solution_str, re.DOTALL)
    if solution is None:
        return None
    else:
        final_answer = solution.group(1).strip()
        return final_answer


def compute_score(solution_str, ground_truth):
    """Compute binary correctness score for code understanding task."""
    answer = extract_solution(solution_str)
    if answer is None:
        # No answer found in the correct format
        return 0.0
    else:
        # Check if the answer matches the ground truth (case-sensitive exact match)
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0


class CodeUnderstandingEnv(BaseTextEnv):
    """
    Environment for code understanding tasks.
    Single-turn environment that extracts the final answer using regex
    and gives a binary correctness reward.
    """

    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward(self, action: str) -> float:
        """Calculate reward based on the model's response."""
        return compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step in the environment."""
        done = True  # Single-turn environment, always done after one step
        reward = self._get_reward(action)

        # Extract the parsed answer for metadata
        parsed_answer = extract_solution(action)

        # No additional observations in single-turn setup
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={"parsed_answer": parsed_answer}
        )
