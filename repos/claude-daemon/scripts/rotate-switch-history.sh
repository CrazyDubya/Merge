#!/bin/bash
#
# Switch History Rotation Script
# Rotates switch-history entries >7 days to compressed monthly archives
# Pattern: Based on archive-timeline.sh, adapted for switch-history
# Target: Keep hot tier (7 days) small for fast daemon startup
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-${HOME}/.claude/daemon}"
SWITCH_HISTORY="${DAEMON_ROOT}/metrics/switch-history.jsonl"
ARCHIVE_DIR="${DAEMON_ROOT}/metrics/archives"
SEVEN_DAYS_AGO=$(date -u -d '7 days ago' +%Y-%m-%d)
LOCKFILE="${DAEMON_ROOT}/metrics/.rotate-switch-history.lock"

mkdir -p "${ARCHIVE_DIR}"

# Concurrent execution protection
exec 200>"${LOCKFILE}"
if ! flock -n 200; then
    echo "ERROR: Another instance of rotate-switch-history.sh is already running"
    echo "If you're sure no other instance exists, remove: ${LOCKFILE}"
    exit 1
fi

# Backup switch-history before modification
BACKUP="${SWITCH_HISTORY}.backup-$(date +%Y%m%d-%H%M%S)"
cp "${SWITCH_HISTORY}" "${BACKUP}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backed up switch-history to ${BACKUP}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting switch-history rotation (entries older than ${SEVEN_DAYS_AGO})"

# Create temp file for hot tier (recent 7 days)
HOT_TEMP=$(mktemp)
trap 'rm -f ${HOT_TEMP}' EXIT

# Split switch-history: hot tier (recent) vs cold tier (archive)
# Handle potential corruption gracefully
jq -c "select(.timestamp >= \"${SEVEN_DAYS_AGO}\")" "${SWITCH_HISTORY}" 2>/dev/null > "${HOT_TEMP}" || {
    echo "WARNING: Corruption detected, attempting recovery with grep"
    grep -a "\"timestamp\":\"${SEVEN_DAYS_AGO}" "${SWITCH_HISTORY}" > "${HOT_TEMP}" || {
        echo "ERROR: Unable to extract recent entries"
        exit 1
    }
}

# Group old entries by month for archival
jq -c "select(.timestamp < \"${SEVEN_DAYS_AGO}\")" "${SWITCH_HISTORY}" 2>/dev/null | \
while IFS= read -r entry; do
    timestamp=$(echo "${entry}" | jq -r '.timestamp' 2>/dev/null || echo "")

    # Skip malformed entries
    if [[ ! "${timestamp}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
        echo "WARNING: Malformed timestamp skipped: ${timestamp}"
        continue
    fi

    month=$(echo "${timestamp}" | cut -d'T' -f1 | cut -d'-' -f1,2)

    archive_file="${ARCHIVE_DIR}/switch-history-${month}.jsonl"
    echo "${entry}" >> "${archive_file}"
done

# Compress monthly archives
for archive in "${ARCHIVE_DIR}"/switch-history-*.jsonl; do
    # Skip if it's a .gz file or doesn't exist
    [[ "${archive}" == *.gz ]] && continue
    [ ! -f "${archive}" ] && continue

    month=$(basename "${archive}" .jsonl | cut -d'-' -f3,4)

    # Check if already compressed
    if [ ! -f "${archive}.gz" ]; then
        gzip -9 "${archive}"

        # Integrity validation
        if ! gunzip -t "${archive}.gz" 2>/dev/null; then
            echo "ERROR: Compressed archive failed integrity check: ${archive}.gz"
            echo "Attempting recovery from backup..."
            cp "${BACKUP}" "${SWITCH_HISTORY}"
            exit 1
        fi

        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Compressed ${month} archive (integrity verified)"
    else
        # Append to existing compressed archive
        TEMP_ARCHIVE=$(mktemp)
        gunzip -c "${archive}.gz" > "${TEMP_ARCHIVE}"
        cat "${archive}" >> "${TEMP_ARCHIVE}"
        rm "${archive}"
        gzip -9 -c "${TEMP_ARCHIVE}" > "${archive}.gz"
        rm "${TEMP_ARCHIVE}"

        # Integrity validation
        if ! gunzip -t "${archive}.gz" 2>/dev/null; then
            echo "ERROR: Updated archive failed integrity check: ${archive}.gz"
            echo "Attempting recovery from backup..."
            cp "${BACKUP}" "${SWITCH_HISTORY}"
            exit 1
        fi

        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Updated ${month} archive (integrity verified)"
    fi
done

# Atomic operation: verify hot tier before replacing switch-history
hot_lines_before=$(wc -l < "${SWITCH_HISTORY}")
hot_lines_new=$(wc -l < "${HOT_TEMP}")

# Sanity check: hot tier should have fewer entries than original
if [ "${hot_lines_new}" -gt "${hot_lines_before}" ]; then
    echo "ERROR: Hot tier has MORE entries than original switch-history"
    echo "  Original: ${hot_lines_before}, Hot tier: ${hot_lines_new}"
    echo "This suggests a logic error. Aborting to prevent data loss."
    echo "Switch-history preserved at: ${BACKUP}"
    exit 1
fi

# Verify hot tier is valid JSONL (skip if corruption recovery was used)
if ! jq -e . "${HOT_TEMP}" >/dev/null 2>&1; then
    echo "WARNING: Hot tier contains invalid JSONL, using as-is (corruption recovery mode)"
fi

# Replace switch-history with hot tier only (atomic operation)
mv "${HOT_TEMP}" "${SWITCH_HISTORY}"

# Calculate compression stats
hot_size=$(wc -c < "${SWITCH_HISTORY}")
archive_size=$(du -sb "${ARCHIVE_DIR}" 2>/dev/null | cut -f1 || echo 0)
hot_lines=$(wc -l < "${SWITCH_HISTORY}")
archived_count=$((hot_lines_before - hot_lines))

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Rotation complete"
echo "  Hot tier: ${hot_lines} entries, $(numfmt --to=iec ${hot_size})"
echo "  Archived: ${archived_count} entries, $(numfmt --to=iec ${archive_size}) compressed"
echo "  Cutoff: ${SEVEN_DAYS_AGO}"
echo "  Backup: ${BACKUP}"
