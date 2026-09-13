from ChipCliff.researcher_algorithm import generate_queries

def test_generate_queries():
    task_details = {"task": "Research HTML5 best practices"}
    summary = generate_queries(task_details)
    assert "Summary" in summary
