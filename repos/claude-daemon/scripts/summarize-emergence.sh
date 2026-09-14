#!/bin/bash
#
# Emergence Log Summarization Script
# Creates monthly summaries of reflections >30 days old
# Pattern: Voyager (raw logs → distilled knowledge, 10:1 compression)
# Target: Preserve key insights, trait evolution, lessons learned
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-${HOME}/.claude/daemon}"
EMERGENCE_LOG="${DAEMON_ROOT}/memory/emergence-log.md"
SUMMARY_DIR="${DAEMON_ROOT}/memory/emergence-summaries"
THIRTY_DAYS_AGO=$(date -u -d '30 days ago' +%Y-%m-%d)

mkdir -p "${SUMMARY_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting emergence log summarization (entries older than ${THIRTY_DAYS_AGO})"

# Backup emergence log
BACKUP="${EMERGENCE_LOG}.backup-$(date +%Y%m%d-%H%M%S)"
cp "${EMERGENCE_LOG}" "${BACKUP}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backed up emergence log to ${BACKUP}"

# Extract entries by date, create monthly summaries
# This is a prototype - in production, would use LLM to actually summarize
# For now, extract key sections (insights, learnings, decisions)

# Get list of months that need summarization
months=$(grep -oP '## \d{4}-\d{2}-\d{2}T' "${EMERGENCE_LOG}" | \
         cut -d'T' -f1 | cut -d' ' -f2 | cut -d'-' -f1,2 | sort -u)

for month in ${months}; do
    month_start="${month}-01"
    # Calculate if month is >30 days old
    if [[ "${month_start}" < "${THIRTY_DAYS_AGO}" ]]; then
        summary_file="${SUMMARY_DIR}/${month}-summary.md"

        if [ ! -f "${summary_file}" ]; then
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Creating summary for ${month}"

            # Extract all entries for this month (using mktemp for security)
            MONTH_TEMP=$(mktemp)
            trap 'rm -f ${MONTH_TEMP}' EXIT
            awk "/^## ${month}/ {flag=1} /^## [0-9]{4}/ && !/^## ${month}/ {flag=0} flag" "${EMERGENCE_LOG}" > "${MONTH_TEMP}"

            # Create summary structure (prototype - manual extraction)
            cat > "${summary_file}" << EOF
# Emergence Log Summary: ${month}

**Generated**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Period**: ${month}
**Compression**: $(wc -l < "${MONTH_TEMP}") lines → summary

---

## Key Insights

$(grep -A2 "### .*Insight\|### The Core Insight\|### Key.*" "${MONTH_TEMP}" | head -20 || echo "- [No structured insights found]")

---

## Trait Evolution

$(grep -A2 "trait\|evolution\|behavior" "${MONTH_TEMP}" | grep -v "^--$" | head -15 || echo "- [No trait evolution documented]")

---

## Lessons Learned

$(grep -A2 "lesson\|learned\|learning" "${MONTH_TEMP}" | grep -v "^--$" | head -15 || echo "- [No explicit lessons documented]")

---

## Architectural Decisions

$(grep -A2 "architectural\|design\|pattern" "${MONTH_TEMP}" | grep -v "^--$" | head -15 || echo "- [No architectural decisions found]")

---

## Collaboration Patterns

$(grep -A2 "collaboration\|working with\|helped" "${MONTH_TEMP}" | grep -v "^--$" | head -10 || echo "- [No collaboration patterns noted]")

---

**Full detail preserved in**: ${BACKUP}
**Original entry count**: $(grep -c "^## ${month}" "${MONTH_TEMP}" || echo 0)
**Lines compressed**: $(wc -l < "${MONTH_TEMP}") → $(wc -l < "${summary_file}")
EOF
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Created ${summary_file}"
        fi
    fi
done

# Calculate stats
summary_count=$(find "${SUMMARY_DIR}" -name "*-summary.md" | wc -l)
summary_size=$(du -sh "${SUMMARY_DIR}" 2>/dev/null | cut -f1 || echo "0")

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Summarization complete"
echo "  Summaries created: ${summary_count}"
echo "  Summary directory size: ${summary_size}"
echo "  Original emergence log: $(du -sh "${EMERGENCE_LOG}" | cut -f1)"
echo ""
echo "  NOTE: This is prototype keyword extraction. Production version would use LLM"
echo "        for semantic summarization preserving insights while compressing verbosity."
