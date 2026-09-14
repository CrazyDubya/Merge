---
TEMPLATE: deployment
DESCRIPTION: Deploy application to environment
PARAMETERS:
  - environment: Environment name (staging/production, required)
  - version: Version number (required)
---

- [ ] [MAINTAINER] Deploy {{version}} to {{environment}}
  TASK_ID: task-{{timestamp}}-deploy-{{version}}-{{environment}}
  OUTPUT: deployments/{{version}}-{{environment}}.log
  VERIFY: [ -f deployments/{{version}}-{{environment}}.log ] && grep -q "SUCCESS\|COMPLETE" deployments/{{version}}-{{environment}}.log
