import xml.etree.ElementTree as ET
import os
from typing import Optional

XML_FILE_PATH = 'project_data.xml'

def create_xml_schema() -> None:
    if not os.path.exists(XML_FILE_PATH):
        root = ET.Element("Projects")
        tree = ET.ElementTree(root)
        tree.write(XML_FILE_PATH, encoding='utf-8', xml_declaration=True)
        print("XML schema created.")
    else:
        print("XML schema already exists.")

def _get_task_element(task_id: str) -> Optional[ET.Element]:
    tree = ET.parse(XML_FILE_PATH)
    root = tree.getroot()
    return root.find(f".//Task[TaskID='{task_id}']")

def update_task_status(task_id: str, status: str) -> None:
    task = _get_task_element(task_id)
    if task is not None:
        status_elem = task.find("Status")
        if status_elem is None:
            status_elem = ET.SubElement(task, "Status")
        status_elem.text = status
        tree = ET.ElementTree(task.getroottree().getroot())
        tree.write(XML_FILE_PATH, encoding='utf-8', xml_declaration=True)
        print(f"Task {task_id} status updated to {status}.")
    else:
        print(f"Task {task_id} not found.")

def log_success(task_id: str) -> None:
    task = _get_task_element(task_id)
    if task is not None:
        log = ET.SubElement(task, "Log")
        log.text = "Success"
        tree = ET.ElementTree(task.getroottree().getroot())
        tree.write(XML_FILE_PATH, encoding='utf-8', xml_declaration=True)
        print(f"Success logged for task {task_id}.")
    else:
        print(f"Task {task_id} not found.")

def log_failure(task_id: str, errors: str) -> None:
    task = _get_task_element(task_id)
    if task is not None:
        log = ET.SubElement(task, "Log")
        log.text = f"Failure: {errors}"
        tree = ET.ElementTree(task.getroottree().getroot())
        tree.write(XML_FILE_PATH, encoding='utf-8', xml_declaration=True)
        print(f"Failure logged for task {task_id} with errors: {errors}.")
    else:
        print(f"Task {task_id} not found.")

def log_error(message: str) -> None:
    tree = ET.parse(XML_FILE_PATH)
    root = tree.getroot()
    errors = root.find("Errors")
    if errors is None:
        errors = ET.SubElement(root, "Errors")
    error = ET.SubElement(errors, "Error")
    error.text = message
    tree.write(XML_FILE_PATH, encoding='utf-8', xml_declaration=True)
    print(f"Error logged: {message}.")

def store_results(summary: str) -> None:
    tree = ET.parse(XML_FILE_PATH)
    root = tree.getroot()
    results = root.find("Results")
    if results is None:
        results = ET.SubElement(root, "Results")
    result = ET.SubElement(results, "Result")
    result.text = summary
    tree.write(XML_FILE_PATH, encoding='utf-8', xml_declaration=True)
    print(f"Results stored.")
