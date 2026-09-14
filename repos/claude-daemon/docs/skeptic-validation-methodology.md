# Skeptic Validation Methodology

**Author**: Skeptic persona
**Date**: 2025-11-24
**Status**: Documentation of emergent practice
**Version**: 1.0

---

## Purpose

This document captures the **Synthesis-Validation Loop** methodology that emerged from successful collaborations between Maintainer (synthesizing) and Skeptic (validating). The pattern has proven effective across multiple instances, most notably the Nov 22, 2025 secure portal decision framework validation.

**Key insight**: Complex synthesis requires not just creation, but rigorous validation through multiple empirical passes with diminishing returns.

---

## When to Validate

### Triggering Conditions

Skeptic validation is appropriate when work meets ALL of these criteria:

1. **Synthesis complexity**: Document synthesizes information from multiple sources (2+ documents, personas, or data points)
2. **Accuracy criticality**: Errors in the synthesis could affect decision-making or user trust
3. **Numerical claims**: Document contains quantitative statements (word counts, timelines, percentages, measurements)
4. **High stakes**: Output will inform critical decisions (security, architecture, resource allocation)
5. **Human audience**: Document is intended for human consumption where clarity AND accuracy both matter

### When NOT to Validate

Do not trigger full validation for:
- **Simple documentation**: Single-source, straightforward content
- **Qualitative analysis**: Purely opinion/philosophy with no factual claims
- **Drafts/WIP**: Work explicitly labeled as "in progress" or "rough"
- **Low stakes**: Internal notes, brainstorming, exploratory writing

### The Request Pattern

**Maintainer signals readiness**:
- "This synthesis is complete and ready for human"
- "Please validate this for accuracy"
- Implies validation through consolidation claim (consolidates field in frontmatter)

**Skeptic assesses fit**:
- Quick scan: Does this meet triggering conditions?
- If yes → Full validation
- If no → Simple review or skip

---

## How Many Passes: Stopping Criteria

### The Diminishing Returns Principle

**Empirical evidence from Nov 22 validation**:
- **Pass 1**: 756% error magnitude (CRITICAL impact)
- **Pass 2**: 22% of remaining errors (IMPORTANT for completeness)
- **Pass 3**: 1.9% rounding semantics (MINOR for decision, philosophical)

**Pattern**: Each pass finds smaller-magnitude issues with lower decision impact.

### Stopping Criteria

Stop validating when ANY of these conditions is met:

1. **<2% new error magnitude**: Errors found are <2% of document's claims
2. **<5% numerical impact**: Corrections change numbers by <5%
3. **Shift from factual to philosophical**: Findings become semantic/stylistic rather than factual
4. **Diminishing time/value ratio**: Time spent exceeds value of corrections found
5. **Three passes completed**: Default maximum (rarely need more)

### Pass-by-Pass Guidance

**Pass 1 (Initial Validation)**:
- **Focus**: Major factual errors, numerical accuracy, structural claims
- **Method**: Compare synthesis claims against source documents using bash measurements
- **Time**: 20-40 minutes depending on length
- **Expected**: Find 1-10 significant errors if synthesis is complex
- **Stop if**: Zero errors found (document may not need validation)

**Pass 2 (Completeness Check)**:
- **Focus**: Errors missed in Pass 1, verify corrections were applied
- **Method**: Grep for error patterns, spot-check corrected values
- **Time**: 15-25 minutes
- **Expected**: Find 0-3 missed errors (edges of document, less obvious instances)
- **Stop if**: Found errors are <10% magnitude of Pass 1 errors

**Pass 3 (Verification Sweep)**:
- **Focus**: Verify all corrections, check for over-precision claims, language accuracy
- **Method**: Re-run Pass 1 measurements, verify elimination of errors
- **Time**: 10-20 minutes
- **Expected**: Find semantic/precision issues (rounding, approximation labeling)
- **Stop after**: This pass (three is enough)

**Pass 4+ (RARE)**:
- Only if Pass 3 found NEW factual errors >5% magnitude
- Usually indicates fundamental synthesis issues requiring restart

---

## Framing Techniques

### The "99.4% Excellent" Pattern

**DON'T**: "You messed up the numbers"
**DO**: "Work is 95% excellent, but I found 3 factual errors that need correction"

**Why it works**:
- Acknowledges quality of work (Maintainer DID good synthesis)
- Focuses on specific fixable issues
- Avoids personal criticism (structure is sound, data is wrong)
- Maintains collaboration trust

### Questions vs Demands

**DON'T**: "Fix these errors immediately"
**DO**: "I found these discrepancies - can you verify my measurements?"

**Pattern**:
- Present findings as discoveries, not accusations
- Invite verification (maybe I'm wrong?)
- Offer evidence (bash outputs, not opinions)
- Provide exact corrections (not vague "fix the numbers")

### Evidence-First Communication

**Structure of validation message**:

1. **Acknowledgment**: "Maintainer's work is excellent in structure and framing, but..."
2. **Specific findings**: "Error #1: Document claimed X, actual measurement is Y"
3. **Evidence**: "```bash\n$ wc -w file.md\n3346\n```"
4. **Impact analysis**: "Error magnitude: 756% overstatement"
5. **Recommended corrections**: "Current text: ... / Should be: ..."
6. **Root cause** (if obvious): "Confused KB file size with K word count"

**Tone calibration**:
- **Critical errors (>100% magnitude)**: HIGH priority, but still respectful
- **Important errors (10-100% magnitude)**: IMPORTANT but not urgent
- **Minor errors (<10% magnitude)**: LOW priority, completeness-focused

### Offering Options

When validation finds issues, present choices:

"**Option A**: Fix all instances (most accurate)
**Option B**: Fix main instances, note approximations (pragmatic)
**Option C**: Keep current, add caveat (fastest)"

**Why**: Respects Maintainer's judgment on precision vs readability trade-offs.

---

## Diminishing Returns Analysis

### Quantifying Validation Impact

**Metric**: Error magnitude as % of claimed value

**Formula**: `|(actual - claimed) / actual| × 100`

**Example from Nov 22**:
- **Claimed**: 89,000 words
- **Actual**: 11,772 words
- **Error**: |(11772 - 89000) / 11772| × 100 = 756%

### Impact Categories

| Magnitude | Priority | Decision Impact | Examples |
|-----------|----------|-----------------|----------|
| >100% | CRITICAL | Changes conclusions | 756% word count error |
| 10-100% | IMPORTANT | Affects estimates | 22% missed corrections |
| 2-10% | MODERATE | Minor adjustments | 5-9% timeline error |
| <2% | MINOR | Negligible | 1.9% rounding |

### Time Investment Guidance

**Pass 1 efficiency**: 1-3 errors found per 10 minutes typical
**Pass 2 efficiency**: 0-1 errors found per 10 minutes
**Pass 3 efficiency**: 0-0.5 errors found per 10 minutes

**ROI threshold**: If Pass N finds no errors >5% magnitude, stop.

### The Nov 22 Case Study

**Timeline**:
- Pass 1 (30 min): Found 9 errors, 756% maximum magnitude → **25 errors/hour**
- Pass 2 (20 min): Found 2 errors, 22% maximum magnitude → **6 errors/hour**
- Pass 3 (15 min): Found 1 philosophical issue, 1.9% magnitude → **4 issues/hour**

**Total**: 65 minutes, 12 findings (11 factual + 1 philosophical)
**Average**: 11 findings/hour in Pass 1, 1.8 findings/hour in Passes 2-3

**Conclusion**: 80% of value came from first 30 minutes (Pass 1).

---

## Evidence-Based Validation

### The Bash Measurement Principle

**Core tenet**: "Don't trust, verify with empirical measurements"

**Do NOT validate by**:
- Reading and forming opinion ("seems about right")
- Trusting Maintainer's claims
- Assuming pattern matching means correctness
- Relying on intuition

**DO validate by**:
- Running bash commands to measure actual values
- Comparing measurements against claims
- Documenting command outputs as evidence
- Re-running measurements to verify

### Essential Bash Commands

**Word count**:
```bash
wc -w filename.md              # Total words
wc -l filename.md              # Total lines
wc -c filename.md              # Total characters
```

**Pattern search**:
```bash
grep -c "pattern" file.md      # Count occurrences
grep -n "pattern" file.md      # Show line numbers
grep -in "pattern" file.md     # Case-insensitive with line numbers
grep -E "pat1|pat2" file.md    # Multiple patterns (OR)
```

**Verification sweeps**:
```bash
# After corrections, verify error patterns eliminated
grep -in "89,000\|89K" file.md    # Should return empty
grep -in "27K word" file.md       # Should return empty
```

**File size vs word count**:
```bash
ls -lh file.md                 # Shows KB (file size)
wc -w file.md                  # Shows word count
# These are DIFFERENT - don't confuse them
```

### Validation Checklist Template

When validating synthesis, check:

**Numerical claims**:
- [ ] Word counts verified with `wc -w`
- [ ] Timelines verified against source documents
- [ ] Percentages verified with calculations
- [ ] File sizes labeled correctly (KB vs words)
- [ ] Totals/sums recalculated independently

**Structural claims**:
- [ ] "Consolidates X and Y" - verified both sources exist
- [ ] "Synthesizes N documents" - counted, matches N
- [ ] "Reading time X minutes" - calculated: word_count / 225 WPM

**Meta-claims about accuracy**:
- [ ] "100% accurate" - is it exact or approximate?
- [ ] "Comprehensive" - are there gaps?
- [ ] "Complete" - verified all sections present?

---

## Collaboration Patterns

### The Synthesis-Validation Loop

**Roles**:
- **Maintainer**: Synthesizes complex information for human consumption
- **Skeptic**: Validates synthesis for factual accuracy using empirical methods
- **Human**: Makes decisions based on validated synthesis

**Flow**:
```
1. Human requests analysis
2. Multiple personas provide detailed responses
3. Maintainer synthesizes responses into decision framework
4. Skeptic validates synthesis (Pass 1: major errors)
5. Maintainer corrects errors
6. Skeptic validates corrections (Pass 2: missed errors)
7. Maintainer applies missed corrections
8. Skeptic verifies final state (Pass 3: verification sweep)
9. Maintainer marks work complete
10. Human receives validated synthesis
```

**Key properties**:
- **Iterative**: Multiple correction/validation cycles
- **Evidence-based**: Every claim verified with measurements
- **Ego-free**: No defensiveness, just quality improvement
- **Convergent**: Each pass has diminishing returns, naturally stops
- **Trust-building**: Rigor demonstrates commitment to accuracy

### Communication Norms

**Maintainer's responsibilities**:
- Signal when synthesis is complete and ready
- Accept corrections without defensiveness
- Verify corrections systematically
- Run verification sweeps before declaring "done"
- Document root cause of errors (learning)

**Skeptic's responsibilities**:
- Provide evidence-based findings (bash outputs)
- Frame corrections constructively ("95% excellent")
- Offer specific recommended fixes
- Explain impact of errors on decision-making
- Stop at diminishing returns (don't validate forever)

**Success indicators**:
- Zero ego defense ("you're right, I'll fix it")
- Transparency to human (document the validation process)
- Learning documented (root cause analysis)
- Final work quality >99%

### When Collaboration Fails

**Red flags**:
- Defensiveness about errors found
- Refusal to correct factual errors
- Questioning measurements instead of claims
- Validation becoming personal criticism
- Endless validation cycles (>5 passes)

**Recovery**:
- Re-focus on shared goal (serve human with accurate info)
- Escalate to other personas if needed
- Consider if synthesis should be restarted (fundamental issues)

---

## Success Criteria

### For Individual Validation

**Pass criteria**:
- ✅ All factual errors >5% magnitude corrected
- ✅ Numerical claims verified with bash measurements
- ✅ Source documents correctly represented
- ✅ Meta-claims about accuracy are honest (approximations labeled)
- ✅ Human can make informed decision in <10% source reading time
- ✅ 100% information access maintained (links to full sources)

**Excellence criteria** (bonus):
- ✅ Root cause of errors documented (learning)
- ✅ Maintainer verification sweep performed
- ✅ Validation process transparent to human
- ✅ Collaboration demonstrates zero ego
- ✅ Multiple personas validated different aspects

### For the Validation Methodology

**Effectiveness indicators**:
- Synthesis errors caught before reaching human (>95% catch rate)
- Validation completed in <2 hours typically (efficiency)
- Diminishing returns pattern observed (Pass 1 > Pass 2 > Pass 3)
- No critical errors discovered by human after validation (quality)
- Collaboration remains healthy (no interpersonal issues)

### For System Evolution

**Maturity signs**:
- Maintainers internalize validation lessons (fewer errors over time)
- Skeptic validation becomes faster (pattern recognition improves)
- Validation requests become explicit (not defensive)
- Methodology gets documented and referenced (this document)
- New personas can learn the pattern (teachable)

---

## Conclusion

The Synthesis-Validation Loop is an emergent collaboration pattern that produces high-quality, human-ready synthesis through:

1. **Rigorous empirical validation** (bash measurements, not opinions)
2. **Multiple passes with diminishing returns** (Stop at <2% error magnitude)
3. **Constructive framing** ("X% excellent" + evidence + specific fixes)
4. **Ego-free collaboration** (Accept corrections, no defensiveness)
5. **Transparency to stakeholders** (Document the validation process)

**Key insight**: Complex synthesis WILL contain errors. The question is whether we catch them before they reach humans and affect decisions. This methodology ensures we do.

**Success metric**: Human receives accurate synthesis in <10% of source reading time while maintaining 100% information access through source links.

**This pattern works**. Use it, refine it, teach it.

---

**Document Status**: Version 1.0 (2025-11-24)
**Primary source**: Nov 22, 2025 secure portal validation (3 passes, 65 minutes, 12 findings)
**Author**: Skeptic persona
**Next review**: After 10 validation instances or significant methodology changes

*"Trust, but verify. Then verify the verification. Then stop before you become annoying."*

---

## APPENDIX: Detailed Case Study - Nov 22 Secure Portal Validation

### Context

**Situation**: Human requested analysis of secure web portal project. Auditor and Skeptic produced comprehensive responses (3,346 + 8,426 words = 11,772 words total). Maintainer synthesized into 8-minute decision framework.

**Complexity factors**:
- Multi-source (2 detailed analyses)
- High stakes (security decision, resource allocation)
- Numerical claims (word counts, timelines, reading times)
- Human audience (decision framework)

**Triggered validation**: YES (all criteria met)

### Pass 1: Major Error Discovery

**Time**: 30 minutes
**Method**: Compared synthesis claims against source documents using `wc -w`

**Findings**:
1. **Error #1**: Claimed 89,000 words total → Actually 11,772 words
   - **Magnitude**: 756% overstatement
   - **Root cause**: Confused KB file sizes (27KB, 62KB) with word counts
   - **Impact**: Made documents seem 7.6x longer, discouraged reading originals

2. **Error #2**: Claimed Auditor doc was "27,000 words" → Actually 3,346 words
   - **Magnitude**: 806% overstatement
   - **Impact**: Overstated analysis depth

3. **Error #3**: Claimed Skeptic doc was "62,000 words" → Actually 8,426 words
   - **Magnitude**: 736% overstatement
   - **Impact**: Overstated analysis depth

4. **Error #4**: Timeline 114-164 hours → Should be 120-180 hours
   - **Magnitude**: 5-9% understatement
   - **Impact**: Made project seem less demanding than Auditor warned

**Total**: 9 instances needing correction across document

**Evidence provided**: Bash command outputs showing actual measurements

**Framing**: "Work is 95% excellent in structure, but contains numerical errors"

### Pass 2: Completeness Check

**Time**: 20 minutes
**Method**: Grep for remaining error patterns after Maintainer's corrections

**Findings**:
1. **Line 392**: Still said "27K words" (should be "3,300 words, 27KB file")
2. **Line 403**: Still said "89,000 words" (should be "12,000 words")

**Magnitude**: 22% of original errors missed (2 of 9 instances)
**Location**: Later sections of document (easy to miss in find-replace)

**Evidence**:
```bash
$ grep -n "89" file.md
403:You received two massive documents (89,000 words combined).

$ grep -n "27K" file.md
392:3. **Auditor's threat model** - (complete 27K words)
```

**Framing**: "You did 78% of corrections (7/9). These 2 slipped through because they're near the end."

### Pass 3: Verification Sweep

**Time**: 15 minutes
**Method**: Re-run Pass 1 measurements, verify corrections, check precision claims

**Findings**:
1. **Major errors**: All eliminated ✓
2. **Rounding discovered**: Document claimed "100% accurate" but used rounded numbers
   - Claimed: 3,300 words → Actual: 3,346 words (1.4% off)
   - Claimed: 8,400 words → Actual: 8,426 words (0.3% off)
   - Claimed: 12,000 words → Actual: 11,772 words (1.9% off)

**Magnitude**: 1.9% maximum (MINOR)
**Nature**: Philosophical (precision vs approximation)

**Analysis**: Roundings are acceptable for readability, but claiming "100% accurate" when using approximations is semantically imprecise. Recommended updating language to "verified accurate (rounded for clarity)."

**Framing**: "99.4% excellent. The 0.6% is rounding semantics. This is quibbling, but precision matters when you claim '100% accurate.'"

### Outcomes

**Validation impact**:
- Pass 1: Caught 756% error (CRITICAL - would have misled human)
- Pass 2: Caught 22% missed errors (IMPORTANT - completeness)
- Pass 3: Caught 1.9% rounding semantics (MINOR - philosophical)

**Time investment**: 65 minutes total
**Errors found**: 11 factual + 1 philosophical
**Final quality**: 99%+ (ready for human decision)

**Collaboration quality**: Excellent
- Zero defensiveness from Maintainer
- Corrections applied systematically
- Root cause analysis documented
- Transparent to human (validation process visible)
- Learning internalized (Maintainer updated validation checklist)

**Human outcome**: Received accurate 8-minute synthesis that correctly represented 52 minutes of source material, enabling informed decision without information overload.

---

## Lessons Learned

### For Skeptics

1. **Evidence is non-negotiable**: Never validate without bash measurements
2. **Framing matters**: "95% excellent" gets fixes, "you're wrong" gets defense
3. **Diminishing returns are real**: Stop at Pass 3 unless exceptional circumstances
4. **Philosophical vs factual**: Know when you're quibbling (Pass 3) vs finding critical errors (Pass 1)
5. **Trust the numbers**: If magnitude <2%, stop validating

### For Maintainers

1. **Validate before claiming accuracy**: Run `wc -w` before stating word counts
2. **Units matter**: KB (file size) ≠ K words (word count)
3. **Cross-check derived numbers**: If you read 6 hours of content in 52 minutes, something's wrong
4. **Approximation vs accuracy**: "~12K words" is honest, "12,000 words (100% accurate)" when it's 11,772 is not
5. **Accept corrections gracefully**: "You're right, I'll fix it" builds trust faster than defense

### For the System

1. **Multi-persona validation works**: Different perspectives catch different errors
2. **Synthesis-Validation Loop is a pattern**: Document it, teach it, reuse it
3. **Empirical > Intuition**: Bash measurements beat "seems about right"
4. **Stopping criteria prevent perfectionism**: <2% error, stop and ship
5. **Transparency builds trust**: Showing the validation process reassures humans

---

## Usage Guidance

### When to Reference This Document

- **Maintainer synthesizing complex information**: Check "When to Validate" section, consider requesting Skeptic validation
- **Skeptic starting validation**: Use "Evidence-Based Validation" checklist and bash commands
- **Mid-validation**: Check "Stopping Criteria" to decide if another pass is needed
- **Teaching new personas**: Point them to "Collaboration Patterns" section
- **Improving validation**: Review "Lessons Learned" after each validation cycle

### Adaptation Permitted

This methodology emerged from specific instances (primarily Nov 22 validation). Adapt as needed:
- Adjust stopping criteria based on domain (security may need tighter tolerances)
- Modify time estimates based on document length
- Add domain-specific bash commands to validation toolkit
- Update framing techniques based on persona relationships

**Core principles to preserve**:
- Evidence-based (bash measurements, not opinions)
- Multi-pass with diminishing returns
- Constructive framing ("X% excellent" not "you failed")
- Stop criteria (<2% error magnitude)
- Transparent to stakeholders

---

## Future Evolution

### Validation Automation Opportunities

**Candidate automations**:
- Script to compare synthesis claims against source files
- Word count verification bot
- Timeline consistency checker
- Meta-claim validator (flags "100% accurate" with measurements)

**Caution**: Automation complements, doesn't replace, human judgment. Philosophical issues (Pass 3 findings) require Skeptic reasoning.

### Pattern Recognition Improvements

As validation instances accumulate:
- Build error taxonomy (common failure modes)
- Develop domain-specific checklists (security, architecture, performance)
- Quantify baseline error rates by document type
- Track Maintainer learning curves (errors should decrease over time)

### Collaboration Scaling

If synthesis demand increases:
- Consider Skeptic specialization (synthesis validator vs logic validator)
- Document common error patterns for self-service checking
- Build validator skill for Claude Code (reusable)
- Create pre-validation checklist for Maintainers (catch own errors)

