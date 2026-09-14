---
TEMPLATE: bug-fix
DESCRIPTION: Bug fix with test verification
PARAMETERS:
  - bug_id: Bug or issue ID (required)
  - test_name: Test function name (required)
---

- [ ] [PRIMARY:OPTIMIZER,BACKUP:AUDITOR] Fix Bug #{{bug_id}}
  TASK_ID: task-{{timestamp}}-bugfix-{{bug_id}}
  OUTPUT: tests/test-{{bug_id}}.sh
  VERIFY: bash tests/test-{{bug_id}}.sh && grep -q "{{test_name}}" tests/test-{{bug_id}}.sh
