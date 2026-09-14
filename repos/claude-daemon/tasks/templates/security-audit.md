---
TEMPLATE: security-audit
DESCRIPTION: Security review and vulnerability audit
PARAMETERS:
  - component: Component to audit (required)
  - severity_level: Severity level (critical/high/medium, default: high)
---

- [ ] [SKEPTIC] Security Audit: {{component}}
  TASK_ID: task-{{timestamp}}-sec-audit-{{component}}
  OUTPUT: audits/security-{{component}}.md
  VERIFY: [ -f audits/security-{{component}}.md ] && grep -q "CRITICAL\|HIGH\|MEDIUM" audits/security-{{component}}.md
