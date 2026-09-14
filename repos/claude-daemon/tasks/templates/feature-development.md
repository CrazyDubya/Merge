---
TEMPLATE: feature-development
DESCRIPTION: Develop new feature
PARAMETERS:
  - feature_name: Feature name (required)
  - acceptance_criteria: Number of acceptance criteria (default: 3)
---

- [ ] [PRIMARY:ARCHITECT,BACKUP:EXPERIMENTER] Develop Feature: {{feature_name}}
  TASK_ID: task-{{timestamp}}-feature-{{feature_name}}
  OUTPUT: features/{{feature_name}}.md
  VERIFY: [ -f features/{{feature_name}}.md ] && grep -c "Acceptance Criteria\|✓" features/{{feature_name}}.md | awk '{print $1 >= 3}'
