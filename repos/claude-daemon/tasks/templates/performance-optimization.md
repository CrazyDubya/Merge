---
TEMPLATE: performance-optimization
DESCRIPTION: Performance improvement task
PARAMETERS:
  - target: Performance target (required)
  - improvement_percent: Target improvement percentage (default: 20)
---

- [ ] [OPTIMIZER] Optimize {{target}} Performance
  TASK_ID: task-{{timestamp}}-perf-{{target}}
  OUTPUT: metrics/benchmark-{{target}}.txt
  VERIFY: [ -f metrics/benchmark-{{target}}.txt ] && grep -q "improved" metrics/benchmark-{{target}}.txt
