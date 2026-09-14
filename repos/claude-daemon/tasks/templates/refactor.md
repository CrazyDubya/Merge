---
TEMPLATE: refactor
DESCRIPTION: Code refactoring with test coverage
PARAMETERS:
  - component: Component or file to refactor (required)
  - min_coverage: Minimum test coverage percentage (default: 80)
---

- [ ] [ARCHITECT] Refactor {{component}}
  TASK_ID: task-{{timestamp}}-refactor-{{component}}
  OUTPUT: {{component}}.refactored
  VERIFY: [ -f {{component}}.refactored ] && grep -q "test" {{component}}.refactored
