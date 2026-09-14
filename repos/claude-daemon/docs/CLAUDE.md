# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory contains a collection of 12 Claude Code Agent Skills located in `~/.claude/skills/`. These skills extend Claude's capabilities for common software development tasks.

## Skills Architecture

All skills follow a consistent structure:

```
~/.claude/skills/<skill-name>/
├── SKILL.md           # Main skill definition with YAML frontmatter
├── templates/         # Reusable templates (optional)
├── scripts/           # Helper scripts (optional)
├── reference/         # Reference documentation (optional)
└── examples/          # Code examples (optional)
```

### SKILL.md Format

Each SKILL.md must include:
- **YAML frontmatter** with `name`, `description`, and optionally `allowed-tools`
- **Instructions section** explaining when and how Claude should use the skill
- **Best practices** and framework-specific guidance

The `description` field is critical - it determines when Claude autonomously activates the skill based on user requests.

## Available Skills

### Code Quality & Style
1. **code-style-enforcer**: Style consistency beyond automated linters
2. **test-coverage-analyzer**: Coverage gap analysis and test suggestions
3. **performance-profiler**: Identifies bottlenecks, N+1 queries, memory leaks

### Documentation & API
4. **api-documentation-generator**: OpenAPI/Swagger doc generation from routes

### Database & Migrations
5. **database-migration-helper**: Creates migrations for Prisma, Sequelize, Alembic, etc.

### DevOps & Infrastructure
6. **docker-optimizer**: Dockerfile best practices, security, multi-stage builds
7. **configuration-validator**: Validates env vars and config files

### Security & Dependencies
8. **dependency-audit-assistant**: Security audits, license compliance, outdated packages

### Internationalization & Accessibility
9. **internationalization-helper**: Extracts strings, manages translation files
10. **accessibility-auditor**: WCAG 2.1 AA compliance checks

### Error Tracking & Git
11. **error-tracking-integrator**: Sentry/Rollbar integration setup
12. **git-workflow-enforcer**: Conventional commits, branch naming, PR templates

## Modifying Skills

### Testing Skill Activation

Skills are automatically discovered by Claude from:
1. `~/.claude/skills/` (personal skills)
2. `.claude/skills/` (project-specific skills)
3. Plugin-bundled skills

Test activation by using trigger keywords from the skill's `description` field in conversation.

### Adding New Skills

1. Create directory: `mkdir -p ~/.claude/skills/<skill-name>`
2. Create SKILL.md with required frontmatter:
   ```yaml
   ---
   name: skill-name-format
   description: What it does and when to use it (max 1024 chars)
   allowed-tools: Read, Grep, Glob  # Optional: restrict tool access
   ---
   ```
3. Add detailed instructions in markdown below frontmatter
4. Test by mentioning relevant keywords in conversation

### Skill Naming Conventions

- **name**: lowercase-with-hyphens, max 64 characters
- **description**: Must explain BOTH what the skill does AND when to use it
- **directory name**: Should match the `name` field

### Tool Restrictions

Use `allowed-tools` to limit skill permissions:
- **Read-only operations**: `Read, Grep, Glob`
- **File modifications**: Add `Write, Edit`
- **Command execution**: Add `Bash`

Example for security-focused skills that should only analyze:
```yaml
allowed-tools: Read, Grep, Glob
```

## Common Patterns

### Template Files
Store reusable templates in `templates/` subdirectory. Reference them in SKILL.md instructions with relative paths.

### Helper Scripts
Executable scripts in `scripts/` should:
- Be marked executable: `chmod +x scripts/*.sh`
- Handle errors gracefully
- Support common package managers/frameworks

### Reference Documentation
Place comprehensive reference material in `reference/` to keep SKILL.md concise and actionable.

### Framework Detection
Skills supporting multiple frameworks should:
1. Detect framework via config files (use Glob)
2. Read existing patterns (use Grep)
3. Apply framework-specific logic

## Debugging Skills

**Skill not activating?**
1. Check `description` specificity - needs clear trigger words
2. Verify file paths: `~/.claude/skills/<name>/SKILL.md`
3. Validate YAML syntax (proper `---` delimiters, no tabs)
4. Run `claude --debug` for error visibility

**Multiple skills conflicting?**
- Use distinct trigger terms in descriptions
- Differentiate use cases clearly

## Key Files

- `~/.claude/skills/*/SKILL.md` - All skill definitions
- `~/.claude/skills/*/templates/` - Reusable templates
- `~/.claude/skills/*/scripts/` - Helper automation
- `~/.claude/skills/*/reference/` - Reference documentation
