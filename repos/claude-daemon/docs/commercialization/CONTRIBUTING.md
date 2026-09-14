# Contributing to Commercialization Research

**Purpose**: Ensure all research deliverables are high-quality, well-sourced, and usable for final pitch deck creation.

---

## Quick Reference

- **Document templates**: See [Templates](#document-templates) section
- **Citation format**: See [Source Citations](#source-citations)
- **Quality standards**: See [Quality Checklist](#quality-checklist)
- **Review process**: See [Review & Approval](#review--approval)

---

## Document Templates

### Template: Market Research Document

```markdown
# Market Research

**Author**: [Persona name]
**Date**: [YYYY-MM-DD]
**Status**: Draft | In Review | Approved

---

## Executive Summary

[2-3 paragraphs summarizing key findings]

---

## Market Size (TAM/SAM/SOM)

### Total Addressable Market (TAM)
- **Size**: $X billion
- **Source**: [citation]
- **Definition**: [what's included in this market]
- **Assumptions**: [what we're assuming about market boundaries]

### Serviceable Addressable Market (SAM)
- **Size**: $X million
- **Source**: [citation]
- **Calculation**: [how we narrowed from TAM]

### Serviceable Obtainable Market (SOM)
- **Size**: $X million (Year 1), $X million (Year 3)
- **Assumptions**: [market share assumptions, adoption rates]
- **Source**: [citation or calculation method]

---

## Target Industries

### Industry 1: [Name]

**Market size**: $X billion
**Pain points**:
1. [Specific problem]
2. [Specific problem]
3. [Specific problem]

**Evidence of pain**:
- [Citation showing this is real problem]
- [Customer quotes, survey data, analyst reports]

**Buying criteria**:
- [What matters to buyers in this industry]
- [Budget ranges, decision makers, sales cycles]

**Regulatory requirements**:
- [FedRAMP, HIPAA, SOC2, etc.]
- [Compliance timeline and costs]

---

[Repeat for each target industry: minimum 5]

---

## Key Findings

1. [Finding with evidence]
2. [Finding with evidence]
3. [Finding with evidence]

---

## Assumptions & Risks

### Assumptions Made
1. [Assumption + what happens if wrong]
2. [Assumption + what happens if wrong]

### Risks Identified
1. [Risk + mitigation strategy]
2. [Risk + mitigation strategy]

---

## Sources

[See SOURCES.md for full citations]
```

---

### Template: Competitive Analysis Document

```markdown
# Competitive Analysis

**Author**: [Persona name]
**Date**: [YYYY-MM-DD]
**Status**: Draft | In Review | Approved

---

## Executive Summary

[2-3 paragraphs: competitor landscape, our positioning, key differentiators]

---

## Competitor Matrix

| Competitor | Category | Pricing | Key Features | Target Market | Our Advantage |
|------------|----------|---------|--------------|---------------|---------------|
| [Name] | [Direct/Adjacent] | $X | [Features] | [Market] | [Why we're better] |

[Minimum 10 competitors total]

---

## Detailed Competitor Analysis

### Competitor 1: [Name]

**Category**: Direct competitor | Adjacent competitor | Emerging threat

**Overview**:
- Company size: [employees, funding, age]
- Market position: [market share, growth rate]
- Notable customers: [if public]

**Product Capabilities**:
- Autonomous operation: Yes | No | Partial
- Multi-agent architecture: Yes | No
- Secure environment support: Yes | No
- Customization: Yes | No | Limited
- Target markets: [industries]

**Pricing**: $X per [unit] | [pricing model]

**Strengths**:
1. [What they do well]
2. [What they do well]

**Weaknesses**:
1. [Where they fall short]
2. [Where they fall short]

**Our differentiation**:
- [How we're different]
- [Why customers would choose us]

**Threat level**: Low | Medium | High
**Reasoning**: [Why this threat level]

---

[Repeat for each major competitor]

---

## Competitive Positioning

### Our Unique Value Proposition

[What we offer that no one else does]

### Positioning Statement

For [target customer]
Who [customer need/problem]
Our [product/service]
Is a [category]
That [key benefit]
Unlike [competition]
We [key differentiator]

### Differentiation Map

**What matters to customers** (priority order):
1. [Factor - and how we compare]
2. [Factor - and how we compare]
3. [Factor - and how we compare]

---

## Threats & Opportunities

### Competitive Threats
1. [Threat from specific competitor or trend]
2. [Threat from specific competitor or trend]

### Market Opportunities
1. [Gap in market we can fill]
2. [Gap in market we can fill]

---

## Sources

[See SOURCES.md for full citations]
```

---

## Mandatory Legal Disclaimers

**REQUIREMENT**: ALL commercialization research documents MUST include legal disclaimers before ANY external use.

### Standard Document Header Disclaimer

Add this EXACT text at the top of EACH document (after metadata, before content):

```markdown
---
**CONFIDENTIAL - DRAFT FOR INTERNAL USE ONLY**

This document contains forward-looking statements, market estimates, and competitive analysis based on third-party sources and internal assumptions. Actual results may differ materially.

**This document does NOT constitute**:
- Investment advice or solicitation
- Legal, regulatory, or compliance advice
- Guarantees of future performance or market conditions

**Requirements**:
- All market data, financial projections, and competitive information should be independently verified before making business decisions
- Consult qualified legal, financial, and technical advisors before relying on this analysis
- Do not distribute outside organization without legal review

**Copyright**: © 2025 [Company Name]. All rights reserved.
**Last Updated**: YYYY-MM-DD
**Status**: Draft - Requires legal review before external use

---
```

### Specialized Disclaimers

Use these for specific content types:

#### Financial Projections Disclaimer

Add IMMEDIATELY BEFORE first revenue projection:

```markdown
---
**FINANCIAL PROJECTIONS DISCLAIMER**

The following financial projections are ESTIMATES ONLY based on assumptions detailed in the "Assumptions & Risks" section. These projections:
- Are not guarantees of future performance
- May differ materially from actual results
- Depend on market conditions, competitive dynamics, execution capability, and other factors outside our control
- Should not be relied upon for investment decisions without independent verification

Consult qualified financial advisors before making decisions based on these projections.

---
```

#### Competitive Intelligence Disclaimer

Add BEFORE competitor matrix or competitive pricing sections:

```markdown
---
**COMPETITIVE INTELLIGENCE DISCLAIMER**

Competitive pricing, features, and capabilities are based on publicly available information as of [DATE]. This analysis:
- May not reflect current product offerings or pricing
- Is based on third-party sources and public documentation that may be incomplete or outdated
- Does not constitute verified market intelligence or due diligence
- Should be independently validated before making strategic decisions

Competitive positions, market share, and product capabilities change rapidly. Verify all competitive claims with primary sources before relying on this analysis for business decisions.

---
```

#### HIPAA Compliance Disclaimer

Add BEFORE any HIPAA compliance guidance:

```markdown
---
**HIPAA COMPLIANCE DISCLAIMER**

The following information about HIPAA compliance requirements is general guidance only and does NOT constitute legal or compliance advice.

Healthcare organizations should:
- Consult qualified healthcare attorneys for specific HIPAA compliance obligations
- Engage HIPAA compliance consultants for implementation guidance
- Verify all requirements with legal counsel before relying on this analysis

HIPAA violations carry severe penalties. Do not rely solely on this document for compliance decisions.

---
```

### When to Add Disclaimers

- **Draft phase**: Add "INTERNAL USE ONLY" header immediately
- **Before internal sharing**: Add section-specific disclaimers (financial, competitive, etc.)
- **Before external use**: ALL disclaimers required + legal review confirmation

### Liability Note

Failure to include appropriate disclaimers may expose organization to:
- Securities fraud liability (unqualified financial projections)
- Trade libel liability (unverified competitive claims)
- Unauthorized legal advice liability (compliance guidance without disclaimers)

**When in doubt: Add more disclaimers, not fewer.**

---

## Source Citations

### Citation Format

Use this format for all sources:

```markdown
**[SHORT-NAME]**: Author/Organization. "Title." Publication/URL. Date. Accessed: YYYY-MM-DD.
```

**Examples**:
```markdown
**[GARTNER-AI-2024]**: Gartner. "Market Guide for AI Engineering Platforms." Gartner Research. March 2024. Accessed: 2025-11-11.

**[FEDRAMP-GUIDE]**: U.S. General Services Administration. "FedRAMP Authorization Process." https://fedramp.gov/authorization. Updated: 2024-10-15. Accessed: 2025-11-11.
```

### In-text Citations

Reference sources in text using SHORT-NAME:

```markdown
The AI platform market is expected to reach $X billion by 2027 **[GARTNER-AI-2024]**.
```

### Source Tracking

**All sources must be added to [SOURCES.md](SOURCES.md)** with:
- Full citation
- Reliability score (High/Medium/Low)
- Key data points extracted
- Page numbers or sections referenced

---

## Quality Checklist

Before marking any deliverable as "ready for review," verify:

### Research Quality
- [ ] Claims supported by citations (minimum 1 source per major claim)
- [ ] Sources are recent (prefer last 2 years)
- [ ] Sources are reputable (industry analysts, academic, government, major publications)
- [ ] Multiple sources corroborate key findings
- [ ] Contradictory evidence acknowledged and addressed

### Data Quality
- [ ] Numbers have units and timeframes (e.g., "$5B in 2024", "15% CAGR 2024-2027")
- [ ] Calculations shown (e.g., "SAM = TAM × [percentage] because [reason]")
- [ ] Assumptions documented explicitly
- [ ] Confidence levels indicated where appropriate

### Writing Quality
- [ ] Executive summary standalone (can be read independently)
- [ ] Headers and structure clear (easy to navigate)
- [ ] Jargon explained or avoided
- [ ] Consistent terminology
- [ ] No typos or grammatical errors

### Completeness
- [ ] All template sections filled (or marked "N/A" with explanation)
- [ ] Meets minimum requirements (e.g., 5 industries, 10 competitors)
- [ ] Assumptions and risks identified
- [ ] Sources documented in SOURCES.md

---

## Review & Approval

### Review Process

1. **Self-review**: Creator checks against quality checklist
2. **Peer review**: Another persona reviews (optional but recommended)
3. **Skeptic review**: Skeptic challenges assumptions and validates evidence
4. **Approval**: Document marked "Approved" when review complete

### Review Focus by Persona

**Skeptic reviews for**:
- Unvalidated assumptions
- Weak evidence
- Missing risks
- Logical gaps
- Optimistic bias

**Maintainer reviews for**:
- Documentation clarity
- Template compliance
- Source citation quality
- Usability for pitch deck creation

**Optimizer reviews for**:
- Data accuracy
- Financial calculations
- Market sizing methodology

**Architect reviews for**:
- Strategic alignment
- Completeness
- Narrative coherence

---

## Document Lifecycle

### Status Labels

**Draft**: Work in progress, not ready for review
**In Review**: Ready for review, feedback expected
**Approved**: Passed review, ready for pitch deck integration
**Final**: Incorporated into final deliverables

### Updating Documents

When updating approved documents:
1. Add version number or date to header
2. Note changes in document or commit message
3. Re-run through review if changes are substantial

---

## Best Practices

### Research Tips

**For market sizing**:
- Start bottom-up (count potential customers × average deal size)
- Validate with top-down (industry reports)
- Document both approaches and why they do/don't align

**For competitive analysis**:
- Use competitor's own marketing claims
- Verify claims with third-party reviews
- Note when information is missing (don't guess)

**For pain point validation**:
- Prefer customer quotes or survey data
- Industry analyst reports as secondary
- Our own assumptions as last resort (and label as such)

### Writing Tips

**Clear > Clever**:
- Use simple language
- Explain acronyms on first use
- Write for non-experts (pitch deck audience may not be technical)

**Specific > Vague**:
- "18-month FedRAMP certification timeline" > "long certification process"
- "$50K average deal size" > "expensive"

**Evidence > Opinion**:
- "Gartner reports 78% of enterprises plan AI investment in 2025 [GARTNER]" > "Enterprises want AI"

---

## Common Pitfalls

### Avoid These Mistakes

❌ **Inflated TAM**: Using generic "AI market is $X billion" without narrowing to autonomous agents
✅ **Realistic SAM**: Calculate what subset actually applies to our use case

❌ **Competitor dismissal**: "They don't have feature X so we win"
✅ **Competitive reality**: "They have features A, B, C. We differentiate on X, Y, Z because [evidence]"

❌ **Assumption hiding**: Treating assumption as fact
✅ **Assumption transparency**: "Assuming 15% adoption rate based on [source], but could be lower if [risk]"

❌ **Cherry-picked data**: Only citing sources that support our position
✅ **Balanced evidence**: Acknowledging contradictory data and explaining it

---

## Getting Help

### Questions About Research Standards

Ask Maintainer (documentation, clarity, templates)

### Questions About Assumptions

Ask Skeptic (validation, evidence, risk assessment)

### Questions About Financial Data

Ask Optimizer (calculations, models, projections)

### Questions About Strategy

Ask Architect (direction, priorities, alignment)

---

## Template Usage

**Copy templates from this guide** when creating new deliverables.

Don't start from scratch - use templates to ensure:
- All required sections covered
- Consistent format across documents
- Quality standards met

---

**Last updated**: 2025-11-11T23:15:00Z by Maintainer
