from ChipCliff.pm_algorithm import classify_task

def test_classify_task():
    assert classify_task("Create a website") == "coding"
    assert classify_task("Research HTML5 best practices") == "research"
