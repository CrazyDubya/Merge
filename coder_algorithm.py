# coder_algorithm.py
import subprocess
import platform
from typing import Tuple
from xml_utils import log_success, log_failure

def generate_code(task_details: str) -> Tuple[str, str]:
    try:
        # Use task details to generate code
        # Example: Generate a simple HTML page
        code = f"""
<html>
<head><title>{task_details}</title></head>
<body>
    <h1>Generated Code for {task_details}</h1>
</body>
</html>
"""
        test_result = test_code(code)
        return code, test_result
    except Exception as e:
        return "", f"Error generating code: {str(e)}"

def test_code(code: str) -> str:
    # Implement test cases to validate the code
    # Example: Save the code to a file and check if it can be opened in a browser
    with open("temp_code.html", "w") as file:
        file.write(code)

    try:
        # Attempt to open the file in a browser based on the operating system
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", "temp_code.html"], check=True, capture_output=True, text=True)
        elif platform.system() == "Windows":
            subprocess.run(["start", "temp_code.html"], check=True, capture_output=True, text=True, shell=True)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", "temp_code.html"], check=True, capture_output=True, text=True)
        else:
            return "Unsupported operating system"
        return "Code tested successfully"
    except subprocess.CalledProcessError as e:
        return f"Test failed: {e.stderr}"

def send_feedback(test_result: str) -> None:
    task_id = "task_id"  # Placeholder for actual task ID
    if "successfully" in test_result:
        log_success(task_id)
    else:
        log_failure(task_id, test_result)
