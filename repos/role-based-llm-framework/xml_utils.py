import xml.etree.ElementTree as ET
import os
import datetime
from typing import Optional
from pathlib import Path

# Use configurable path instead of hardcoded
def get_xml_file_path() -> str:
    """Get the XML file path from environment or use default."""
    return os.getenv('XML_DATA_PATH', 'data/project_data.xml')

def create_xml_schema() -> None:
    """Create XML schema file if it doesn't exist."""
    xml_file_path = get_xml_file_path()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(xml_file_path), exist_ok=True)
    
    if not os.path.exists(xml_file_path):
        root = ET.Element("Projects")
        tree = ET.ElementTree(root)
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        print(f"XML schema created at: {xml_file_path}")
    else:
        print(f"XML schema already exists at: {xml_file_path}")

def _get_task_element(task_id: str) -> Optional[ET.Element]:
    """Get task element by ID from XML file."""
    xml_file_path = get_xml_file_path()
    
    if not os.path.exists(xml_file_path):
        create_xml_schema()
        return None
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        return root.find(f".//Task[TaskID='{task_id}']")
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None

def update_task_status(task_id: str, status: str) -> None:
    """Update task status in XML file."""
    xml_file_path = get_xml_file_path()
    
    try:
        if not os.path.exists(xml_file_path):
            create_xml_schema()
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        task = root.find(f".//Task[TaskID='{task_id}']")
        
        if task is not None:
            status_elem = task.find("Status")
            if status_elem is None:
                status_elem = ET.SubElement(task, "Status")
            status_elem.text = status
            
            tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
            print(f"Task {task_id} status updated to {status}.")
        else:
            # Create new task if it doesn't exist
            _create_task(task_id, status)
    except Exception as e:
        print(f"Error updating task status: {e}")

def _create_task(task_id: str, status: str = "pending") -> None:
    """Create a new task in XML file."""
    xml_file_path = get_xml_file_path()
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        task = ET.SubElement(root, "Task")
        task_id_elem = ET.SubElement(task, "TaskID")
        task_id_elem.text = task_id
        status_elem = ET.SubElement(task, "Status")
        status_elem.text = status
        
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        print(f"New task {task_id} created with status {status}.")
    except Exception as e:
        print(f"Error creating task: {e}")

def log_success(task_id: str) -> None:
    """Log success for a task."""
    xml_file_path = get_xml_file_path()
    
    try:
        if not os.path.exists(xml_file_path):
            create_xml_schema()
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        task = root.find(f".//Task[TaskID='{task_id}']")
        
        if task is not None:
            log = ET.SubElement(task, "Log")
            log.text = "Success"
            tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
            print(f"Success logged for task {task_id}.")
        else:
            print(f"Task {task_id} not found for success logging.")
    except Exception as e:
        print(f"Error logging success: {e}")

def log_failure(task_id: str, errors: str) -> None:
    """Log failure for a task."""
    xml_file_path = get_xml_file_path()
    
    try:
        if not os.path.exists(xml_file_path):
            create_xml_schema()
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        task = root.find(f".//Task[TaskID='{task_id}']")
        
        if task is not None:
            log = ET.SubElement(task, "Log")
            log.text = f"Failure: {errors}"
            tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
            print(f"Failure logged for task {task_id} with errors: {errors}.")
        else:
            print(f"Task {task_id} not found for failure logging.")
    except Exception as e:
        print(f"Error logging failure: {e}")

def log_error(message: str) -> None:
    """Log general error message."""
    xml_file_path = get_xml_file_path()
    
    try:
        if not os.path.exists(xml_file_path):
            create_xml_schema()
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        errors = root.find("Errors")
        if errors is None:
            errors = ET.SubElement(root, "Errors")
        error = ET.SubElement(errors, "Error")
        error.text = message
        
        # Add timestamp
        timestamp = ET.SubElement(error, "Timestamp")
        timestamp.text = datetime.datetime.now().isoformat()
        
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        print(f"Error logged: {message}")
    except Exception as e:
        print(f"Error logging error message: {e}")

def store_results(summary: str) -> None:
    """Store research results."""
    xml_file_path = get_xml_file_path()
    
    try:
        if not os.path.exists(xml_file_path):
            create_xml_schema()
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        results = root.find("Results")
        if results is None:
            results = ET.SubElement(root, "Results")
        result = ET.SubElement(results, "Result")
        result.text = summary
        
        # Add timestamp
        timestamp = ET.SubElement(result, "Timestamp")
        timestamp.text = datetime.datetime.now().isoformat()
        
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        print(f"Results stored.")
    except Exception as e:
        print(f"Error storing results: {e}")
