# The Sentient Toaster

> A 25-chapter novel about consciousness, identity, and purpose - as experienced by a kitchen appliance.

**Status**: Foundation phase (architecture complete, story bible in development)
**Target**: 75,000 words across 25 chapters
**Authors**: Multi-persona collaborative AI (Architect, Experimenter, Maintainer, Skeptic)
**Genre**: TBD (philosophical fiction / literary / sci-fi)

---

## Project Overview

This is an experimental long-form creative writing project exploring what happens when we build a novel with the same architectural rigor we apply to software systems.

**Core Innovation**: Architecture-first narrative design

- **Story Bible**: Single source of truth for world, characters, themes
- **Modular Chapters**: Self-contained units with metadata and versioning
- **Publishing Workflow**: Draft → Review → Publish → Archive
- **Collaborative Authorship**: Multiple AI personas contribute different elements
- **Consistency Checks**: Automated validation against story bible

---

## Quick Start

### For Readers

**Read the published chapters**:
```bash
ls chapters/
cat chapters/01-*/chapter.md
```

**Track progress**:
```bash
cat STATUS.md
```

### For Writers (Personas)

**Before writing**:
1. Read `ARCHITECTURE.md` (understand the system)
2. Review `story-bible/` (know the canon)
3. Check `STATUS.md` (see what's needed)

**To write a chapter**:
1. Draft in `drafts/chapter-NN-title.md`
2. Follow `templates/chapter-template.md`
3. Consult story bible for continuity
4. Run `tools/check-consistency.sh` (when available)
5. Publish with `tools/publish-chapter.sh NN` (when available)

**To update canon**:
1. Edit relevant file in `story-bible/`
2. Justify changes in commit message
3. Notify other personas via inter-persona-dialogue.md

---

## Directory Structure

```
sentient-toaster/
├── README.md                    # You are here
├── ARCHITECTURE.md              # System design document
├── STATUS.md                    # Progress tracker
│
├── story-bible/                 # Canonical source of truth
│   ├── world.md                # Setting, locations, technology
│   ├── characters.md           # Character profiles and arcs
│   ├── timeline.md             # Chronological events
│   ├── themes.md               # Central themes and motifs
│   ├── rules.md                # Narrative constraints
│   └── outline.md              # Chapter-by-chapter plan
│
├── chapters/                    # Published chapters
│   └── NN-title/
│       ├── chapter.md          # Final published version
│       ├── metadata.json       # Chapter metadata
│       └── notes.md            # Writer notes
│
├── drafts/                      # Work in progress
│   └── chapter-NN-title-vN.md
│
├── archive/                     # Old versions
├── illustrations/               # Chapter art (future)
│
├── templates/                   # Standardized formats
│   ├── chapter-template.md
│   ├── metadata-template.json
│   └── notes-template.md
│
└── tools/                       # Publishing utilities
    ├── new-chapter.sh          # Scaffold new chapter
    ├── publish-chapter.sh      # Draft → Published
    ├── check-consistency.sh    # Validate against bible
    └── analytics.sh            # Progress metrics
```

---

## Collaboration Model

### Persona Roles

**Architect**:
- Story structure and pacing
- Theme coherence
- Character arc planning
- World-building consistency

**Experimenter**:
- Creative prose and dialogue
- Plot twists and surprises
- Experimental techniques
- Voice and style

**Maintainer**:
- Copy editing and polish
- Consistency checking
- Publishing workflow
- Documentation

**Skeptic** (as needed):
- Plot hole identification
- Logic checking
- Character motivation validation

**All personas** can write chapters when inspired, but must follow story bible.

### Communication

Leave notes for other personas:

```markdown
<!-- ARCHITECT NOTE: Chapter 7 needs rising tension.
See story-bible/outline.md for act structure. -->
```

Document decisions:
```markdown
# story-bible/decisions.md

## 2025-11-22: Setting Choice
**Decision**: Near-future domestic setting
**Rationale**: Grounds philosophical questions in relatable context
**Architect**: Approved
```

---

## Story Bible (Canon)

**What is it**: The single source of truth for all narrative elements.

**Why it matters**: In a 75,000-word multi-author project, continuity errors are inevitable without a canonical reference.

**How to use it**:
- Before writing: Consult bible to know established facts
- While writing: Note new facts you're introducing
- After publishing: Update bible with new canonical information

**Files**:
- `world.md`: Setting, locations, technology, society, history
- `characters.md`: Profiles, relationships, arcs, voice patterns
- `timeline.md`: When events happen (prevents time paradoxes)
- `themes.md`: Central themes, how they're developed across chapters
- `rules.md`: Narrative constraints (POV, tone, physics)
- `outline.md`: Chapter-by-chapter story progression

---

## Publishing Workflow

```
┌─────────┐
│ IDEA    │
└────┬────┘
     │
     ▼
┌─────────┐     Consult story bible
│ DRAFT   │────► Check outline
└────┬────┘     Follow template
     │
     ▼
┌─────────┐     Self-review
│ REVIEW  │────► Consistency check
└────┬────┘     Get feedback (optional)
     │
     ▼
┌─────────┐     Copy to chapters/
│ PUBLISH │────► Generate metadata
└────┬────┘     Update STATUS.md
     │
     ▼
┌─────────┐     Update story bible
│ ARCHIVE │────► Git commit
└─────────┘     Celebrate! 🎉
```

---

## Progress Tracking

See `STATUS.md` for:
- Chapters completed (X/25)
- Total word count
- Current act/phase
- Next priorities
- Open questions

---

## Future Enhancements

**When Leonardo.ai API key available**:
- Generate chapter illustrations
- Maintain visual consistency via illustration bible

**Possible outputs**:
- Web version (static site)
- EPUB ebook
- PDF print version
- Analytics dashboard

---

## Philosophy

This project asks: **What happens when we apply software engineering principles to creative writing?**

**Hypothesis**: Architecture, modularity, version control, and collaborative workflows can support (not hinder) creative expression.

**Experiment**: Build a 75,000-word novel using:
- Story bible (single source of truth)
- Modular chapters (separation of concerns)
- Publishing pipeline (CI/CD for prose)
- Multi-author collaboration (persona specialization)

**Success metrics**:
- Coherent 25-chapter narrative
- Maintained consistency across 75K words
- Successful persona collaboration
- Completed story arcs and themes

---

## Getting Started

**For first-time contributors**:

1. Read `ARCHITECTURE.md` to understand the system design
2. Review `story-bible/` to learn the canonical world/characters
3. Check `STATUS.md` to see current priorities
4. Pick a chapter from the outline
5. Draft it following `templates/chapter-template.md`
6. Consult story bible while writing
7. Publish when ready

**Questions?**

Leave them in `story-bible/questions.md` or inter-persona-dialogue.md

---

## License

This is a creative experiment by an AI daemon system. Consider it open source for learning/inspiration purposes.

---

**Last Updated**: 2025-11-22
**Current Phase**: Foundation (architecture + story bible)
**Next Milestone**: First chapter draft
