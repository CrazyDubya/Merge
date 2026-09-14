---
TEMPLATE: api-endpoint
DESCRIPTION: Implement REST API endpoint
PARAMETERS:
  - endpoint_path: API path (e.g., /users, required)
  - method: HTTP method (GET/POST/PUT/DELETE, default: GET)
---

- [ ] [ARCHITECT] Implement API: {{method}} {{endpoint_path}}
  TASK_ID: task-{{timestamp}}-api-{{endpoint_path}}
  OUTPUT: src/routes{{endpoint_path}}.js
  VERIFY: [ -f src/routes{{endpoint_path}}.js ] && grep -q "{{method}}" src/routes{{endpoint_path}}.js
