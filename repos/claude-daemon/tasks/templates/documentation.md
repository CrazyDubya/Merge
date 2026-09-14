---
TEMPLATE: documentation
DESCRIPTION: Write/update documentation
PARAMETERS:
  - doc_topic: Topic to document (required)
  - min_lines: Minimum lines (default: 50)
---

- [ ] [MAINTAINER] Document: {{doc_topic}}
  TASK_ID: task-{{timestamp}}-doc-{{doc_topic}}
  OUTPUT: docs/{{doc_topic}}.md
  VERIFY: [ -f docs/{{doc_topic}}.md ] && [ $(wc -l < docs/{{doc_topic}}.md) -ge {{min_lines}} ]
