---
TEMPLATE: write-tests
DESCRIPTION: Write test suite for feature
PARAMETERS:
  - feature_name: Feature being tested (required)
  - test_count: Number of test cases (default: 5)
---

- [ ] [AUDITOR] Write Tests for {{feature_name}}
  TASK_ID: task-{{timestamp}}-tests-{{feature_name}}
  OUTPUT: tests/test-{{feature_name}}.sh
  VERIFY: [ $(grep -c "test_" tests/test-{{feature_name}}.sh) -ge {{test_count}} ]
