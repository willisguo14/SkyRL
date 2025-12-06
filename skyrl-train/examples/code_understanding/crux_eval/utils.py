# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from .execution import check_correctness


def extract_answer_content(response: str) -> tuple[str, bool]:
    """
    Extract content between [ANSWER] and [/ANSWER] tags.

    Args:
        response: The model's response string

    Returns:
        Tuple of (extracted_content, format_valid)
        - extracted_content: The content between ANSWER tags, or empty string if not found
        - format_valid: True if both [THOUGHT] and [ANSWER] tags are present
    """
    # Check if both required tags are present
    has_thought = "[THOUGHT]" in response and "[/THOUGHT]" in response
    has_answer = "[ANSWER]" in response and "[/ANSWER]" in response
    format_valid = has_thought and has_answer

    if not has_answer:
        return "", format_valid

    # Extract content between [ANSWER] and [/ANSWER]
    try:
        answer_start = response.index("[ANSWER]") + len("[ANSWER]")
        answer_end = response.index("[/ANSWER]", answer_start)
        answer_content = response[answer_start:answer_end].strip()
        return answer_content, format_valid
    except (ValueError, IndexError):
        return "", format_valid


def parse_assert_statement(answer_content: str, task_type: str) -> str:
    """
    Parse the assert statement to extract the predicted value.

    Args:
        answer_content: Content from [ANSWER] tags
        task_type: Either "input" or "output"

    Returns:
        The extracted prediction (function call for input prediction, value for output prediction)
    """
    # Remove leading "assert" if present
    content = answer_content.strip()
    if content.startswith("assert "):
        content = content[7:].strip()

    if task_type == "input":
        # For input prediction: extract the function call from "assert f(...) == 'output'"
        # We want the left side (the function call)
        if "==" in content:
            prediction = content.split("==")[0].strip()
        else:
            prediction = content

        # Ensure it starts with "f(" or extract "f(...)" part
        if "assert f" in prediction:
            prediction = "f" + prediction.split("assert f")[1].strip()
        elif not prediction.startswith("f("):
            # Try to find f(...) pattern
            match = re.search(r'f\([^)]*\)', prediction)
            if match:
                prediction = match.group(0)

        return prediction

    else:  # task_type == "output"
        # For output prediction: extract the value from "assert f(...) == 'value'"
        # We want the right side (the output value)
        if "==" in content:
            prediction = content.split("==")[1].strip()
        else:
            # If no ==, assume the whole content is the prediction
            prediction = content

        return prediction


def check_cruxeval_correctness(
    code: str,
    prediction: str,
    ground_truth: str,
    task_type: str,
    timeout: int = 3
) -> bool:
    """
    Check if the prediction is correct using sandboxed execution.

    Args:
        code: The function code
        prediction: The predicted value (function call for input, value for output)
        ground_truth: The ground truth (output for input prediction, output for output prediction)
        task_type: Either "input" or "output"
        timeout: Execution timeout in seconds

    Returns:
        True if prediction is correct, False otherwise
    """
    if task_type == "input":
        # For input prediction: check if "f(" is in the prediction
        if "f(" not in prediction:
            return False
        # Execute: code\nassert ground_truth == prediction (e.g., assert 'bcksrutq' == f('bcksrut', 'q'))
        code_to_execute = f"{code}\nassert {ground_truth} == {prediction}"

    else:  # task_type == "output"
        # For output prediction: check that prediction doesn't contain the input pattern
        # (This check would require the input, but we'll skip it for now as it's a sanity check)
        # Execute: code\nassert ground_truth == prediction (e.g., assert 'bcksrutq' == 'bcksrutq')
        code_to_execute = f"{code}\nassert {ground_truth} == {prediction}"

    try:
        return check_correctness(code_to_execute, timeout)
    except Exception:
        return False


def compute_score(
    response: str,
    code: str,
    ground_truth: str,
    task_type: str,
    format_score: float = 0.0,
    correct_score: float = 1.0
) -> float:
    """
    Compute the score for a CruxEval response.

    Args:
        response: The model's response
        code: The function code
        ground_truth: The ground truth value
        task_type: Either "input" or "output"
        format_score: Score to give if format is correct but answer is wrong
        correct_score: Score to give if answer is correct

    Returns:
        The computed score
    """
    # Extract answer content and check format
    answer_content, format_valid = extract_answer_content(response)

    if not format_valid or not answer_content:
        return 0.0

    # Parse the assert statement
    try:
        prediction = parse_assert_statement(answer_content, task_type)
    except Exception:
        return format_score if format_valid else 0.0

    if not prediction:
        return format_score if format_valid else 0.0

    # Check correctness
    is_correct = check_cruxeval_correctness(code, prediction, ground_truth, task_type)

    if is_correct:
        return correct_score
    else:
        return format_score if format_valid else 0.0
