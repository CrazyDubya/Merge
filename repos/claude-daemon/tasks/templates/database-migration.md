---
TEMPLATE: database-migration
DESCRIPTION: Database schema migration
PARAMETERS:
  - table_name: Table name (required)
  - migration_type: Type (create/alter/drop, required)
---

- [ ] [MAINTAINER] Database: {{migration_type}} {{table_name}}
  TASK_ID: task-{{timestamp}}-db-{{table_name}}
  OUTPUT: migrations/{{migration_type}}-{{table_name}}.sql
  VERIFY: [ -f migrations/{{migration_type}}-{{table_name}}.sql ] && grep -q "{{table_name}}" migrations/{{migration_type}}-{{table_name}}.sql
