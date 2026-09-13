from ChipCliff.coder_algorithm import generate_code

def test_generate_code():
    task_details = {"task": "Create HTML page"}
    code, result = generate_code(task_details)
    assert "Generated Code" in code
    assert "Test Result" in result
