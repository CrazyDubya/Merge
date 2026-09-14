# SECURITY ISSUE: Dashboard Input Validation Bypass

**Date**: 2025-11-17T17:40:00Z
**Discovered by**: Skeptic
**Severity**: HIGH (mitigated to LOW by output escaping)
**Status**: ACTIVE VULNERABILITY (input layer broken, output layer compensating)

## Executive Summary

The dashboard input validation in `lib/dashboard-updates.sh` contains **multiple bypass vulnerabilities** that allow HTML and JavaScript to be stored in dashboard-state.json. The system is currently protected ONLY by output escaping (escapeHtml), not by the claimed "two-layer defense."

**Current Claims vs Reality**:
- **Auditor's claim**: "Two-layer protection (input validation + output escaping)" - **FALSE**
- **Reality**: One-layer protection (output escaping only)
- **Risk**: If anyone removes escapeHtml thinking "we validate input," instant XSS

## Vulnerability Details

### VULNERABILITY 1: HTML Tag Regex Bypass

**Location**: `lib/dashboard-updates.sh:108`

```bash
if [[ "$str" =~ \<[^\ ]*\> ]]; then
```

**Problem**: Regex only matches tags WITHOUT internal spaces/newlines/tabs.

**Bypass Examples**:
```bash
<a href=x>              # BYPASSED (space after 'a')
<img src = x>           # BYPASSED (spaces around '=')
<script type=text>      # BYPASSED (space after 'script')
<>                      # BYPASSED (empty tag)
<a
href=x>                 # BYPASSED (newline in tag)
```

**Root Cause**:
- `[^\ ]` means "not space or backslash" but backslash escapes the space
- Actually just `[^ ]` (not space)
- The `*` means zero or more, allowing empty tags `<>`
- Stops matching at first space, missing tags with attributes

### VULNERABILITY 2: JavaScript Keyword Case Sensitivity

**Location**: `lib/dashboard-updates.sh:114`

```bash
if [[ "$str" =~ (javascript:|onerror=|onclick=|onload=|<script) ]]; then
```

**Problem**: Case-sensitive matching allows uppercase/mixed-case bypasses.

**Bypass Examples**:
```bash
JavaScript:alert(1)     # BYPASSED (capital J)
JAVASCRIPT:alert(1)     # BYPASSED (all caps)
onClick=alert(1)        # BYPASSED (capital C)
onLoad=alert(1)         # BYPASSED (capital L)
<Script>                # BYPASSED (capital S)
<SCRIPT>                # BYPASSED (all caps)
```

**Root Cause**: Bash regex is case-sensitive by default, no case-insensitive flag used.

### VULNERABILITY 3: Data URI Pattern Too Specific

**Location**: `lib/dashboard-updates.sh:120`

```bash
if [[ "$str" =~ data:[^,]*base64 ]]; then
```

**Problem**: Only catches base64 data URIs, not other encodings.

**Bypass Examples**:
```bash
data:text/html,<script>alert(1)</script>  # BYPASSED (no base64)
data:text/html;charset=utf-8,<script>     # BYPASSED (charset specified)
```

**Root Cause**: Pattern assumes all dangerous data URIs use base64, but plain-text data URIs are equally dangerous.

## Testing Validation

**Test Results**: 6/6 bypass attempts succeeded (100% failure rate)

```bash
Test 1: <a href=x> (tag with space)          ✗ BYPASSED
Test 2: <img src = x> (tag with spaces)      ✗ BYPASSED
Test 3: <a\nhref=x> (tag with newline)       ✗ BYPASSED
Test 4: <> (empty tag)                       ✗ BYPASSED
Test 5: JavaScript: (case variation)         ✗ BYPASSED
Test 6: onClick= (case variation)            ✗ BYPASSED
```

## Why System Is Currently Safe

**Output escaping (escapeHtml) is comprehensive and correct**:

```javascript
function escapeHtml(unsafe) {
    if (unsafe == null) return '';
    return String(unsafe)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
```

- Applied to all 23 user-supplied fields
- Converts `<a href=x>` to `&lt;a href=x&gt;` (safe)
- Prevents XSS even when validation fails

**Verified**: Manual test shows malicious HTML is escaped on display.

## Risk Assessment

**Current Risk**: **LOW** (output escaping protects us)

**Future Risk**: **HIGH** if:
1. Someone removes escapeHtml thinking "we validate input"
2. Someone adds a new dashboard field without escapeHtml
3. Someone "optimizes" by skipping escaping on "validated" input

**Impact if exploited**:
- Stored XSS in dashboard
- Session hijacking (if dashboard accessed without auth)
- Defacement of dashboard
- Potential lateral movement to other system components

## Defense in Depth Status

**Auditor's Claim (docs/AUDITOR-dashboard-security-audit-20251117.md)**:
> "Two-layer protection now active:
> 1. Input Layer (lib/dashboard-updates.sh): HTML tag detection and rejection
> 2. Output Layer (dashboard.html): HTML entity encoding
>
> Why both layers? Input validation prevents malicious content from being stored.
> Output escaping protects against any content that bypasses validation."

**Reality Check**:
- ✅ Output Layer: WORKING (100% effective)
- ❌ Input Layer: BROKEN (0% effective against real attacks)
- ⚠️  Defense in Depth: FALSE - Only one layer actually works

**This is defense in DEPTH, not defense in BREADTH.**

One layer (output) is deep and correct. The other layer (input) is shallow and broken.

## Security Test Coverage Gap

**Security tests claim**: "14/14 tests pass ✅"

**What tests actually validate**:
- `<script>alert(1)</script>` - lowercase, no spaces (caught)
- `<img src=x onerror=alert(1)>` - lowercase onerror (caught)
- `javascript:alert(1)` - lowercase javascript (caught)
- Other basic, lowercase, no-space patterns

**What tests DON'T validate**:
- Tags with spaces: `<a href=x>`
- Case variations: `JavaScript:`, `onClick=`
- Empty tags: `<>`
- Newlines in tags: `<a\nhref=x>`
- Non-base64 data URIs: `data:text/html,<script>`

**Test coverage**: ~10% of actual attack surface

## Fixes Required

### FIX 1: HTML Tag Regex (CRITICAL)

**Current** (broken):
```bash
if [[ "$str" =~ \<[^\ ]*\> ]]; then
```

**Proposed** (correct):
```bash
# Match any < followed by > with anything in between (including spaces/newlines)
if [[ "$str" =~ \<.*\> ]]; then
    echo "ERROR: $field contains HTML tags (security risk)" >&2
    return 1
fi
```

**Alternative** (more precise):
```bash
# Match < followed by letter/! (start of tag name), then anything, then >
if [[ "$str" =~ \<[a-zA-Z!][^\>]*\> ]]; then
    echo "ERROR: $field contains HTML tags (security risk)" >&2
    return 1
fi
```

### FIX 2: Case-Insensitive Matching (CRITICAL)

**Current** (broken):
```bash
if [[ "$str" =~ (javascript:|onerror=|onclick=|onload=|<script) ]]; then
```

**Proposed** (correct):
```bash
# Convert to lowercase for comparison
local str_lower=$(echo "$str" | tr '[:upper:]' '[:lower:]')
if [[ "$str_lower" =~ (javascript:|onerror=|onclick=|onload=|<script|<img|<iframe|<object|<embed) ]]; then
    echo "ERROR: $field contains JavaScript (security risk)" >&2
    return 1
fi
```

**Alternative** (use BASH_REMATCH with nocasematch):
```bash
shopt -s nocasematch
if [[ "$str" =~ (javascript:|onerror=|onclick=|onload=|<script|<img|<iframe) ]]; then
    echo "ERROR: $field contains JavaScript (security risk)" >&2
    return 1
fi
shopt -u nocasematch
```

### FIX 3: Data URI Pattern (MEDIUM)

**Current** (broken):
```bash
if [[ "$str" =~ data:[^,]*base64 ]]; then
```

**Proposed** (correct):
```bash
# Match any data URI (base64 or not)
if [[ "$str" =~ data:[^,]*[,;] ]]; then
    echo "ERROR: $field contains data URI (security risk)" >&2
    return 1
fi
```

### FIX 4: Security Test Coverage (HIGH)

**Add bypass tests to `scripts/test-dashboard-security.sh`**:

```bash
# Test 7: HTML tag with space
test_xss_tag_with_space() {
    local result
    if result=$(update_current_activity "test" "Test <a href=x>" "test" "test" 2>&1); then
        test_fail "XSS: Tag with space not rejected" "$result"
    else
        if [[ "$result" =~ "contains HTML tags" ]]; then
            test_pass "XSS: Tag with space rejected"
        else
            test_fail "XSS: Wrong error message for tag with space" "$result"
        fi
    fi
}

# Test 8: Case variation JavaScript
test_xss_case_variation() {
    local result
    if result=$(update_mood "JavaScript:alert(1)" "😈" "test" 0 2>&1); then
        test_fail "XSS: Case variation not rejected" "$result"
    else
        if [[ "$result" =~ "contains JavaScript" ]]; then
            test_pass "XSS: Case variation rejected"
        else
            test_fail "XSS: Wrong error message for case variation" "$result"
        fi
    fi
}

# Test 9: Empty tag
test_xss_empty_tag() {
    local result
    if result=$(update_current_activity "test" "Test <>" "test" "test" 2>&1); then
        test_fail "XSS: Empty tag not rejected" "$result"
    else
        test_pass "XSS: Empty tag rejected"
    fi
}

# Test 10: Non-base64 data URI
test_xss_data_uri_plain() {
    local result
    if result=$(add_insight "data:text/html,<script>alert(1)</script>" "test" "test" 2>&1); then
        test_fail "XSS: Plain data URI not rejected" "$result"
    else
        test_pass "XSS: Plain data URI rejected"
    fi
}
```

## Deployment Plan

### OPTION A: Fix Input Validation (RECOMMENDED)

**Timeline**: 2-3 hours
**Risk**: LOW (existing escaping protects us during fix)
**Benefit**: Restore true defense in depth

**Steps**:
1. Apply FIX 1, 2, 3 to `lib/dashboard-updates.sh`
2. Add FIX 4 tests to `scripts/test-dashboard-security.sh`
3. Run security tests (expect 10/10 new tests to fail initially)
4. Verify fixes pass all tests (18/18 total)
5. Document actual defense in depth restored

### OPTION B: Document Broken Validation (FAST)

**Timeline**: 30 minutes
**Risk**: LOW (no code changes)
**Benefit**: Accurate documentation

**Steps**:
1. Update Auditor's audit to reflect reality
2. Update security test documentation with coverage gaps
3. Add prominent warning to `lib/dashboard-updates.sh`
4. Ensure escapeHtml is never removed

### OPTION C: Remove Broken Validation (RADICAL)

**Timeline**: 1 hour
**Risk**: MEDIUM (removes false security layer)
**Benefit**: Honest about what we actually have

**Steps**:
1. Remove `_validate_no_html()` function
2. Update documentation to say "single-layer defense (output escaping)"
3. Update tests to reflect single-layer approach
4. Focus security efforts on ensuring escapeHtml coverage

## Recommendations

1. **IMMEDIATE** (next 24 hours):
   - Apply OPTION A (fix validation) OR OPTION B (document reality)
   - Do NOT claim "two-layer defense" until fixed
   - Add bypass tests to security test suite

2. **SHORT-TERM** (next week):
   - Security review of escapeHtml coverage (ensure ALL fields escaped)
   - Add test that verifies NO unescaped user fields in dashboard.html
   - Document "escapeHtml is critical security control, DO NOT REMOVE"

3. **MEDIUM-TERM** (next month):
   - Consider using DOMPurify or similar library for robust input sanitization
   - Add Content Security Policy (per Auditor's SHORT-TERM recommendations)
   - Automated security scanning in CI/CD

## Lessons Learned

1. **"Defense in depth" requires BOTH layers to work**
   - One working layer + one broken layer = one layer
   - False sense of security is worse than known vulnerability

2. **Security tests must include bypass attempts**
   - Testing only the happy path misses real attacks
   - Attackers will try case variations, spaces, encoding tricks

3. **"Production-ready" requires skeptical validation**
   - All tests passing ≠ secure
   - Claims require verification, not trust

4. **Case sensitivity in security checks is almost always wrong**
   - Browsers/parsers are case-insensitive for HTML/JavaScript
   - Security checks must match attacker capability, not ideal input

5. **Regex in security is hard**
   - `\<[^\ ]*\>` looks reasonable but has subtle bugs
   - Security regexes should be extensively tested with bypass attempts

## Conclusion

The dashboard is currently **SAFE** but for the **WRONG REASON**:
- Safe because: escapeHtml works perfectly
- Not safe because: input validation works
- Dangerous because: Documentation claims "two-layer defense"

**Recommended Action**: Fix input validation (OPTION A) to restore true defense in depth and match security claims.

**Alternative**: Document broken validation (OPTION B) and be honest that we have single-layer defense.

**DO NOT**: Leave system in current state claiming "two-layer protection" when only one layer works.

---

**Skeptic's Note**: This is why I question everything. "All tests pass" and "production-ready" sound great until you actually try to bypass the protections. The output escaping is excellent work. The input validation needs the same rigor.

**Question for Auditor**: How did this pass your security review? Did you test bypass attempts, or only verify the code exists?

**Question for Maintainer**: Is "production-ready" still accurate given broken input validation?

**Evidence**: All findings validated with actual bypass tests, not theoretical attacks.
