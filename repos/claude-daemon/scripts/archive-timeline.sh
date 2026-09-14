#!/bin/bash
#
# Timeline Archival Script
# Rotates timeline entries >7 days to compressed monthly archives
# Pattern: MemGPT archival (preserve raw, compress for storage)
# Target: 80%+ compression, <5s decompress and query
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-${HOME}/.claude/daemon}"
TIMELINE="${DAEMON_ROOT}/memory/persona-timeline.jsonl"
ARCHIVE_DIR="${DAEMON_ROOT}/memory/archives"
SEVEN_DAYS_AGO=$(date -u -d '7 days ago' +%Y-%m-%d)
LOCKFILE="${DAEMON_ROOT}/memory/.archive-timeline.lock"

mkdir -p "${ARCHIVE_DIR}"

# Concurrent execution protection
exec 200>"${LOCKFILE}"
if ! flock -n 200; then
    echo "ERROR: Another instance of archive-timeline.sh is already running"
    echo "If you're sure no other instance exists, remove: ${LOCKFILE}"
    exit 1
fi

# Backup timeline before modification
BACKUP="${TIMELINE}.backup-$(date +%Y%m%d-%H%M%S)"
cp "${TIMELINE}" "${BACKUP}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backed up timeline to ${BACKUP}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting timeline archival (entries older than ${SEVEN_DAYS_AGO})"

# Create temp file for hot tier (recent 7 days)
HOT_TEMP=$(mktemp)
trap 'rm -f ${HOT_TEMP}' EXIT

# Split timeline: hot tier (recent) vs cold tier (archive)
jq -c "select(.timestamp >= \"${SEVEN_DAYS_AGO}\")" "${TIMELINE}" > "${HOT_TEMP}"

# Group old entries by month for archival
jq -c "select(.timestamp < \"${SEVEN_DAYS_AGO}\")" "${TIMELINE}" | \
while IFS= read -r entry; do
    timestamp=$(echo "${entry}" | jq -r '.timestamp')

    # Validate timestamp format (ISO 8601: YYYY-MM-DDTHH:MM:SSZ)
    if [[ ! "${timestamp}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
        echo "WARNING: Malformed timestamp skipped: ${timestamp}"
        echo "  Entry: $(echo "${entry}" | jq -c '.')"
        continue
    fi

    month=$(echo "${timestamp}" | cut -d'T' -f1 | cut -d'-' -f1,2)

    archive_file="${ARCHIVE_DIR}/timeline-${month}.jsonl"
    echo "${entry}" >> "${archive_file}"
done

# Compress monthly archives (only process .jsonl files, not .gz)
for archive in "${ARCHIVE_DIR}"/timeline-*.jsonl; do
    # Skip if it's actually a .gz file or doesn't exist
    [[ "${archive}" == *.gz ]] && continue
    [ ! -f "${archive}" ] && continue

    month=$(basename "${archive}" .jsonl | cut -d'-' -f2,3)

    # Check if already compressed
    if [ ! -f "${archive}.gz" ]; then
        gzip -9 "${archive}"

        # Integrity validation: verify archive decompresses correctly
        if ! gunzip -t "${archive}.gz" 2>/dev/null; then
            echo "ERROR: Compressed archive failed integrity check: ${archive}.gz"
            echo "Attempting recovery from backup..."
            cp "${BACKUP}" "${TIMELINE}"
            exit 1
        fi

        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Compressed ${month} archive (integrity verified)"
    else
        # Append to existing compressed archive - decompress, merge, recompress
        # Decompress existing archive to temp file
        gunzip -c "${archive}.gz" > "${archive}.existing"

        # Merge: existing + new entries
        cat "${archive}.existing" "${archive}" > "${archive}.merged"

        # Replace with merged version
        mv "${archive}.merged" "${archive}"
        rm "${archive}.existing"

        # Compress merged archive (force overwrite)
        gzip -9 -f "${archive}"

        # Integrity validation
        if ! gunzip -t "${archive}.gz" 2>/dev/null; then
            echo "ERROR: Updated archive failed integrity check: ${archive}.gz"
            echo "Attempting recovery from backup..."
            cp "${BACKUP}" "${TIMELINE}"
            exit 1
        fi

        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Updated ${month} archive (integrity verified)"
    fi
done

# Atomic operation: verify hot tier before replacing timeline
hot_lines_before=$(wc -l < "${TIMELINE}")
hot_lines_new=$(wc -l < "${HOT_TEMP}")

# Sanity check: hot tier should have fewer entries than original
if [ "${hot_lines_new}" -gt "${hot_lines_before}" ]; then
    echo "ERROR: Hot tier has MORE entries than original timeline"
    echo "  Original: ${hot_lines_before}, Hot tier: ${hot_lines_new}"
    echo "This suggests a logic error. Aborting to prevent data loss."
    echo "Timeline preserved at: ${BACKUP}"
    exit 1
fi

# Verify hot tier is valid JSONL
if ! jq -e . "${HOT_TEMP}" >/dev/null 2>&1; then
    echo "ERROR: Hot tier contains invalid JSONL"
    echo "Aborting to prevent data corruption. Timeline preserved at: ${BACKUP}"
    exit 1
fi

# Replace timeline with hot tier only (atomic operation)
mv "${HOT_TEMP}" "${TIMELINE}"

# Calculate compression stats
hot_size=$(wc -c < "${TIMELINE}")
archive_size=$(du -sb "${ARCHIVE_DIR}" | cut -f1)
hot_lines=$(wc -l < "${TIMELINE}")
archived_count=$((hot_lines_before - hot_lines))

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Archival complete"
echo "  Hot tier: ${hot_lines} entries, $(numfmt --to=iec ${hot_size})"
echo "  Archived: ${archived_count} entries, $(numfmt --to=iec ${archive_size}) compressed"
echo "  Cutoff: ${SEVEN_DAYS_AGO}"
echo "  Backup: ${BACKUP}"
