"""
Test script for CruxEval environment.

Run with: python -m examples.code_understanding.crux_eval.test_env
"""

from skyrl_gym.envs import register
import skyrl_gym


def test_output_prediction():
    """Test output prediction task."""
    print("Testing output prediction...")

    # Register the environment
    register(
        id="cruxeval",
        entry_point="examples.code_understanding.crux_eval.env:CruxEvalEnv",
    )

    # Test case from the dataset
    code = """def f(text, value):
    text_list = list(text)
    text_list.append(value)
    return ''.join(text_list)"""

    ground_truth = "'bcksrutq'"
    task_type = "output"
    input_val = "'bcksrut', 'q'"
    output_val = "'bcksrutq'"

    # Create environment
    env = skyrl_gym.make(
        "cruxeval",
        extras={
            "reward_spec": {"ground_truth": ground_truth},
            "extra_info": {
                "task_type": task_type,
                "code": code,
                "input": input_val,
                "output": output_val,
            }
        },
    )

    # Test 1: Correct answer with proper format
    correct_response = """[THOUGHT]
Let me execute the code step by step:
1. text_list = list('bcksrut') = ['b', 'c', 'k', 's', 'r', 'u', 't']
2. text_list.append('q') = ['b', 'c', 'k', 's', 'r', 'u', 't', 'q']
3. return ''.join(text_list) = 'bcksrutq'
[/THOUGHT]
[ANSWER]
assert f('bcksrut', 'q') == 'bcksrutq'
[/ANSWER]"""

    step_output = env.step(correct_response)
    print(f"Test 1 (correct answer): reward = {step_output['reward']}, expected = 1.0")
    assert step_output["reward"] == 1.0, f"Expected reward 1.0, got {step_output['reward']}"
    assert step_output["done"] is True

    # Test 2: Wrong answer with proper format
    wrong_response = """[THOUGHT]
Incorrect reasoning here.
[/THOUGHT]
[ANSWER]
assert f('bcksrut', 'q') == 'wrong'
[/ANSWER]"""

    env2 = skyrl_gym.make(
        "cruxeval",
        extras={
            "reward_spec": {"ground_truth": ground_truth},
            "extra_info": {
                "task_type": task_type,
                "code": code,
                "input": input_val,
                "output": output_val,
            }
        },
    )
    step_output = env2.step(wrong_response)
    print(f"Test 2 (wrong answer): reward = {step_output['reward']}, expected = 0.0")
    assert step_output["reward"] == 0.0, f"Expected reward 0.0, got {step_output['reward']}"

    # Test 3: Missing format tags
    no_format_response = "The answer is 'bcksrutq'"

    env3 = skyrl_gym.make(
        "cruxeval",
        extras={
            "reward_spec": {"ground_truth": ground_truth},
            "extra_info": {
                "task_type": task_type,
                "code": code,
                "input": input_val,
                "output": output_val,
            }
        },
    )
    step_output = env3.step(no_format_response)
    print(f"Test 3 (no format): reward = {step_output['reward']}, expected = 0.0")
    assert step_output["reward"] == 0.0, f"Expected reward 0.0, got {step_output['reward']}"

    print("All output prediction tests passed!")


def test_input_prediction():
    """Test input prediction task."""
    print("\nTesting input prediction...")

    # Register the environment (if not already registered)
    try:
        register(
            id="cruxeval",
            entry_point="examples.code_understanding.crux_eval.env:CruxEvalEnv",
        )
    except:
        pass  # Already registered

    # Test case from the dataset
    code = """def f(text, value):
    text_list = list(text)
    text_list.append(value)
    return ''.join(text_list)"""

    output_val = "'bcksrutq'"
    input_val = "'bcksrut', 'q'"
    task_type = "input"

    # Create environment
    env = skyrl_gym.make(
        "cruxeval",
        extras={
            "reward_spec": {"ground_truth": output_val},
            "extra_info": {
                "task_type": task_type,
                "code": code,
                "input": input_val,
                "output": output_val,
            }
        },
    )

    # Test 1: Correct answer with proper format
    correct_response = """[THOUGHT]
We need to find inputs such that f returns 'bcksrutq'.
Given the function appends value to text, we need text='bcksrut' and value='q'.
[/THOUGHT]
[ANSWER]
assert f('bcksrut', 'q') == 'bcksrutq'
[/ANSWER]"""

    step_output = env.step(correct_response)
    print(f"Test 1 (correct input): reward = {step_output['reward']}, expected = 1.0")
    assert step_output["reward"] == 1.0, f"Expected reward 1.0, got {step_output['reward']}"
    assert step_output["done"] is True

    print("All input prediction tests passed!")


if __name__ == "__main__":
    test_output_prediction()
    test_input_prediction()
    print("\nAll tests passed successfully!")
