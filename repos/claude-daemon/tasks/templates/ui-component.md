---
TEMPLATE: ui-component
DESCRIPTION: Build frontend UI component
PARAMETERS:
  - component_name: Component name (required)
  - framework: Framework (React/Vue/Angular, default: React)
---

- [ ] [EXPERIMENTER] Build Component: {{component_name}}
  TASK_ID: task-{{timestamp}}-ui-{{component_name}}
  OUTPUT: src/components/{{component_name}}.jsx
  VERIFY: [ -f src/components/{{component_name}}.jsx ] && grep -q "export\|function\|const" src/components/{{component_name}}.jsx
