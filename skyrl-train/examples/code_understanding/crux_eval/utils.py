import re
from .execution import check_correctness


def extract_answer_content(response: str) -> str:
    """
    Extract content between [ANSWER] and [/ANSWER] tags.

    Required format (with flexible whitespace):
    [THOUGHT] (optional, since it's already in the prompt)
    ...
    [/THOUGHT]
    [ANSWER]
    ...
    [/ANSWER]

    Args:
        response: The model's response string

    Returns:
        The content between ANSWER tags (stripped), or empty string if format is invalid
    """
    # Flexible regex: makes opening [THOUGHT] tag optional since it's in the prompt
    # Allows any whitespace/newlines before/after/between tags
    pattern = r'^\s*(?:\[THOUGHT\]\s*)?(.+?)\s*\[/THOUGHT\]\s*\[ANSWER\]\s*(.+?)\s*\[/ANSWER\]\s*$'
    match = re.match(pattern, response, re.DOTALL)

    if match:
        return match.group(2).strip()
    else:
        return ""


def parse_assert_statement(answer_content: str, task_type: str) -> str:
    """
    Parse the assert statement to extract the predicted value.

    Expected format: "assert f(...) == 'value'" or just "f(...) == 'value'"

    Args:
        answer_content: Content from [ANSWER] tags
        task_type: Either "input" or "output"

    Returns:
        The extracted prediction (function call for input, value for output)
    """
    content = answer_content.strip()

    # Remove "assert " prefix if present
    if content.startswith("assert "):
        content = content[7:].strip()

    # Split on "==" to get left and right sides
    if "==" not in content:
        return ""

    left, right = content.split("==", 1)
    left = left.strip()
    right = right.strip()

    if task_type == "input":
        # For input prediction: return the function call (left side)
        # Validate it looks like a function call
        if not (left.startswith("f(") and left.endswith(")")):
            return ""
        return left
    else:  # task_type == "output"
        # For output prediction: return the value (right side)
        return right


def check_cruxeval_correctness(
    code: str,
    prediction: str,
    task_type: str,
    input_val: str,
    output_val: str,
    timeout: int = 3
) -> bool:
    """
    Check if the prediction is correct using sandboxed execution.

    For input prediction:
        - prediction is f(...) (function call with predicted input)
        - Executes: code + "assert prediction == output_val"
        - Verifies the predicted input produces the expected output

    For output prediction:
        - prediction is a value (predicted output)
        - Executes: code + "assert f(input_val) == prediction"
        - Verifies the given input produces the predicted output

    Args:
        code: The function code
        prediction: The predicted value (function call for input, value for output)
        task_type: Either "input" or "output"
        input_val: The input value from the dataset
        output_val: The output value from the dataset
        timeout: Execution timeout in seconds

    Returns:
        True if prediction is correct, False otherwise
    """
    if task_type == "input":
        # For input prediction: verify predicted function call equals expected output
        code_to_execute = f"{code}\nassert {prediction} == {output_val}"
    else:  # task_type == "output"
        # For output prediction: verify given input produces predicted output
        code_to_execute = f"{code}\nassert f({input_val}) == {prediction}"

    try:
        return check_correctness(code_to_execute, timeout)
    except Exception:
        return False


def compute_score(
    response: str,
    code: str,
    ground_truth: str,
    task_type: str,
    input_val: str,
    output_val: str,
) -> float:
    """
    Compute the score for a CruxEval response.

    Args:
        response: The model's response
        code: The function code
        ground_truth: The ground truth output value
        task_type: Either "input" or "output"
        input_val: The input value from the dataset
        output_val: The output value from the dataset

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Extract answer content (empty if format is invalid)
    answer_content = extract_answer_content(response)
    if not answer_content:
        return 0.0

    # Parse the assert statement
    try:
        prediction = parse_assert_statement(answer_content, task_type)
    except Exception:
        return 0.0

    if not prediction:
        return 0.0

    # Check correctness
    is_correct = check_cruxeval_correctness(
        code, prediction, task_type, input_val, output_val
    )

    return 1.0 if is_correct else 0.0
