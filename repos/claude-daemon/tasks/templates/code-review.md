---
TEMPLATE: code-review
DESCRIPTION: Code review and approval task
PARAMETERS:
  - pr_number: Pull request number (required)
  - min_files: Minimum files to review (default: 1)
---

- [ ] [AUDITOR] Code Review: PR #{{pr_number}}
  TASK_ID: task-{{timestamp}}-code-review-{{pr_number}}
  OUTPUT: reviews/pr-{{pr_number}}-review.md
  VERIFY: [ -f reviews/pr-{{pr_number}}-review.md ] && [ $(wc -l < reviews/pr-{{pr_number}}-review.md) -ge 10 ]
