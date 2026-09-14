---
TEMPLATE: write-chapter
DESCRIPTION: Create a task to write a book chapter
PARAMETERS:
  - chapter_number: Chapter number (required)
  - min_words: Minimum word count (required)
  - persona: Persona to assign (optional, default: EXPERIMENTER)
---

- [ ] [{{persona}}] Write Chapter {{chapter_number}}
  TASK_ID: task-{{timestamp}}-write-chapter-{{chapter_number}}
  OUTPUT: chapters/chapter-{{chapter_number}}.md
  VERIFY: [ $(wc -w < chapters/chapter-{{chapter_number}}.md) -ge {{min_words}} ]
