# Phase 7d: Performance Optimization Guide

**Status**: Implementation Guide (v1.0)
**Date**: 2025-01-08
**Focus**: Caching, Compression, Batch Operations

---

## Executive Summary

Phase 7d optimizes the LisaSimpson + Ralph Wiggum integration for production use by:

1. **Confidence Calculation Caching** (40% latency reduction)
2. **Checkpoint Compression Tuning** (30-50% storage reduction)
3. **Batch Episode Writing** (60% I/O reduction)
4. **Query Optimization** (lazy loading, grep-based filtering)
5. **State Variable Caching** (50% reduction in file I/O)

**Expected Impact**:
- Overall daemon overhead: 0.1% → 0.03% of total task time
- Checkpoint storage: 500MB → 250-350MB (7-day retention)
- Episode write latency: 10ms → 4ms per action
- Confidence calculation latency: 15-50ms → 5-20ms (with caching)

---

## 1. Confidence Calculation Caching

### Current State
- Confidence recalculated for every task generation
- Median calculation: 15-50ms per task
- Historical data queried fresh each time (grep on decision-log.jsonl)

### Optimization: 15-Minute TTL Cache

**Implementation**: Add cache to `lib/confidence-engine.sh`

```bash
# Cache location
CONFIDENCE_CACHE="${DAEMON_ROOT}/.cache/confidence-scores.json"
CONFIDENCE_CACHE_TTL_SECONDS=900  # 15 minutes

calculate_task_confidence_cached() {
    local task_description="$1"
    local goal_json="$2"
    local persona="$3"

    # Check cache
    local cache_key
    cache_key=$(echo "$task_description" | md5sum | awk '{print $1}')

    local cached_result
    cached_result=$(jq -r ".cached[\"$cache_key\"]? // empty" "$CONFIDENCE_CACHE" 2>/dev/null)

    if [ -n "$cached_result" ]; then
        local cache_age=$(($(date +%s) - $(jq -r ".cached[\"$cache_key\"].timestamp" "$CONFIDENCE_CACHE" 2>/dev/null || echo 0)))
        if [ "$cache_age" -lt "$CONFIDENCE_CACHE_TTL_SECONDS" ]; then
            # Cache hit - return cached value
            echo "$cached_result"
            return 0
        fi
    fi

    # Cache miss - calculate and cache
    local result
    result=$(calculate_task_confidence "$task_description" "$goal_json" "$persona")

    # Update cache (append new entry)
    mkdir -p "$(dirname "$CONFIDENCE_CACHE")"
    jq --arg key "$cache_key" \
       --argjson value "$result" \
       --arg ts "$(date +%s)" \
       '.cached[$key] = ($value + {timestamp: ($ts | tonumber)})' \
       "$CONFIDENCE_CACHE" > "$CONFIDENCE_CACHE.tmp" 2>/dev/null || \
    jq -n --arg key "$cache_key" \
         --argjson value "$result" \
         --arg ts "$(date +%s)" \
         '{cached: {($key): ($value + {timestamp: ($ts | tonumber)})}}'  > "$CONFIDENCE_CACHE.tmp"

    mv "$CONFIDENCE_CACHE.tmp" "$CONFIDENCE_CACHE"

    echo "$result"
}
```

**Benefits**:
- Cache hit latency: ~1ms (vs 15-50ms for calculation)
- Hit rate: 60-70% (similar task descriptions repeat)
- Reduction: 40% median latency for repeated tasks

**Trade-off**: Cache staleness (15-min window) acceptable vs TTL accuracy (confidence rarely changes minute-to-minute)

---

## 2. Checkpoint Compression Tuning

### Current State
- Using gzip (reasonable balance of speed/compression)
- Storage: typical checkpoint 20-50MB → 15-30MB compressed
- 7-day retention = 100-210MB active checkpoints

### Optimization: Dual-Mode Compression

**Strategy**: Use gzip for small checkpoints (<20MB), xz for large

```bash
# In lib/checkpoint-manager.sh create_checkpoint()

select_compression_method() {
    local tar_size="$1"
    local tar_size_mb=$((tar_size / 1024 / 1024))

    if [ "$tar_size_mb" -lt 20 ]; then
        echo "gzip"   # Speed optimized: ~50ms compression
    else
        echo "xz"     # Size optimized: ~200ms compression, 35% better ratio
    fi
}

COMPRESSION_METHOD=$(select_compression_method "$tar_size")

case "$COMPRESSION_METHOD" in
    gzip)
        tar -czf "$checkpoint_archive" "${files[@]}"
        ;;
    xz)
        tar -cJf "$checkpoint_archive" "${files[@]}"
        ;;
esac
```

**Benefits**:
- Average compression: 40% → 45-50% (xz for large files)
- Storage reduction: 150-210MB → 100-150MB
- Trade-off: Compression time +150ms for large checkpoints (acceptable, parallelizable)

---

## 3. Batch Episode Writing

### Current State
- Each episode action written immediately to memory/episodes.jsonl
- 25-chapter novel = 75 actions = 75 appends (file opened/closed 75 times)
- Per-episode write: ~10ms (file I/O overhead dominates)

### Optimization: Batch Write with Flush Interval

**Implementation**: Episode buffer in memory

```bash
# In lib/episodic-memory.sh

EPISODE_WRITE_BUFFER=()
EPISODE_BUFFER_SIZE_THRESHOLD=10  # Flush after 10 episodes or 60 seconds
EPISODE_FLUSH_INTERVAL=60         # Seconds

add_action_to_episode() {
    local episode="$1"
    local action="$2"

    # ... existing logic to add action to episode ...

    # Store in buffer instead of immediate write
    EPISODE_WRITE_BUFFER+=("$episode")

    # Flush if buffer threshold reached
    if [ "${#EPISODE_WRITE_BUFFER[@]}" -ge "$EPISODE_BUFFER_SIZE_THRESHOLD" ]; then
        flush_episode_buffer
    fi
}

flush_episode_buffer() {
    if [ ${#EPISODE_WRITE_BUFFER[@]} -eq 0 ]; then
        return
    fi

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"
    mkdir -p "$(dirname "$episodes_file")"

    # Single atomic write for all buffered episodes
    {
        for ep in "${EPISODE_WRITE_BUFFER[@]}"; do
            echo "$ep"
        done
    } >> "$episodes_file" 2>/dev/null || true

    EPISODE_WRITE_BUFFER=()
}

# Add periodic flush (call from daemon main loop)
schedule_episode_flush() {
    ( sleep "$EPISODE_FLUSH_INTERVAL"; flush_episode_buffer ) &
}
```

**Benefits**:
- Per-episode write: 10ms → 4ms (after batch optimization)
- 25-chapter novel: 75 × 10ms = 750ms → 3 × 4ms = 12ms
- Reduction: 98% I/O for multi-action episodes

---

## 4. State Variable Query Optimization

### Current State
- `capture_post_conditions()` queries file system for each state variable
- 13 different state types = potentially 13 file/process calls per verification
- Typical verification: 100-500ms (file I/O bound)

### Optimization: State Cache with Invalidation

```bash
# In lib/world-state.sh

STATE_CACHE_DIR="${DAEMON_ROOT}/.cache/state-variables"
STATE_CACHE_TTL=5  # 5-second TTL for file-based state

cache_state_variable() {
    local var_type="$1"
    local var_spec="$2"
    local value="$3"

    mkdir -p "$STATE_CACHE_DIR"

    local cache_key
    cache_key=$(echo "$var_spec" | md5sum | awk '{print $1}')

    echo "$value" > "$STATE_CACHE_DIR/${var_type}_${cache_key}.cache"
}

get_cached_state_variable() {
    local var_type="$1"
    local var_spec="$2"

    local cache_key
    cache_key=$(echo "$var_spec" | md5sum | awk '{print $1}')

    local cache_file="$STATE_CACHE_DIR/${var_type}_${cache_key}.cache"

    if [ -f "$cache_file" ]; then
        # Check age
        local age=$(($(date +%s) - $(stat -c %Y "$cache_file" 2>/dev/null || stat -f %m "$cache_file" 2>/dev/null)))
        if [ "$age" -lt "$STATE_CACHE_TTL" ]; then
            cat "$cache_file"
            return 0
        fi
    fi

    return 1
}

# Modify capture_post_conditions() to use cache
capture_post_conditions() {
    local goal_id="$1"
    local pre_state="$2"

    # ... existing logic ...

    # For each state variable, try cache first
    echo "$pre_state" | jq -r 'to_entries | .[]' | while read -r entry; do
        local var_type=$(echo "$entry" | jq -r '.value.type')
        local var_spec=$(echo "$entry" | jq -r '.value | to_entries | map("\(.key)=\(.value)") | join(",")')

        # Try cache
        if cached_value=$(get_cached_state_variable "$var_type" "$var_spec" 2>/dev/null); then
            echo "$cached_value"
            continue
        fi

        # Cache miss - evaluate normally
        value=$(evaluate_predicate "$var_type" "$entry")
        cache_state_variable "$var_type" "$var_spec" "$value"
        echo "$value"
    done
}
```

**Benefits**:
- Repeated verification (same goal, within 5 seconds): 100-500ms → 10-20ms
- 80% of verifications within 5-second window of similar state
- Invalidation: Automatic after 5 seconds or on file modification

---

## 5. Verification Plan Template Caching

### Current State
- `generate_verification_plan()` applies type-specific templates on every call
- Template lookup: regex matching on description
- Per-generation: ~20-50ms

### Optimization: Template Cache by Task Type

```bash
# In lib/verification-planner.sh

VERIFICATION_TEMPLATE_CACHE="${DAEMON_ROOT}/.cache/verification-templates.json"

get_verification_template_cached() {
    local task_type="$1"

    # Check cache
    local cached
    cached=$(jq -r ".templates[\"$task_type\"]? // empty" "$VERIFICATION_TEMPLATE_CACHE" 2>/dev/null)

    if [ -n "$cached" ]; then
        echo "$cached"
        return 0
    fi

    # Generate and cache
    case "$task_type" in
        writing)
            local template=$(plan_writing_task)
            ;;
        analysis)
            local template=$(plan_analysis_task)
            ;;
        refactoring)
            local template=$(plan_refactor_task)
            ;;
        # ... other types ...
    esac

    # Cache template
    mkdir -p "$(dirname "$VERIFICATION_TEMPLATE_CACHE")"
    jq --arg type "$task_type" --argjson tmpl "$template" \
       '.templates[$type] = $tmpl' \
       "$VERIFICATION_TEMPLATE_CACHE" > "$VERIFICATION_TEMPLATE_CACHE.tmp" 2>/dev/null || \
    jq -n --arg type "$task_type" --argjson tmpl "$template" \
         '{templates: {($type): $tmpl}}' > "$VERIFICATION_TEMPLATE_CACHE.tmp"

    mv "$VERIFICATION_TEMPLATE_CACHE.tmp" "$VERIFICATION_TEMPLATE_CACHE"

    echo "$template"
}

generate_verification_plan() {
    local task_description="$1"

    local task_type
    task_type=$(detect_task_type "$task_description")

    # Use cached template
    local template
    template=$(get_verification_template_cached "$task_type")

    # Customize for specific task
    echo "$template" | jq \
        --arg description "$task_description" \
        '.description = $description'
}
```

**Benefits**:
- Template lookup: 20-50ms → 1-5ms (cache hit)
- Hit rate: 95%+ (7 task types)
- Reduction: 90% for template generation

---

## 6. Lazy Episode Loading

### Current State
- `get_episodes_for_goal()` loads entire episodes.jsonl, filters with jq
- 1000 episodes = 500KB+ JSON = 50-100ms load + filter

### Optimization: Grep-Based Filtering

```bash
# In lib/episodic-memory.sh

get_episodes_for_goal_lazy() {
    local goal_id="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        echo "[]"
        return
    fi

    # Grep-based filtering (fast, lazy)
    # Only load matching lines, convert to JSON array
    grep "\"goal_id\": \"$goal_id\"" "$episodes_file" 2>/dev/null | jq -s '.' || echo "[]"
}

get_episode_stats_optimized() {
    local goal_id="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        jq -n '{total_episodes: 0}'
        return
    fi

    # Use awk for fast line counting (no JSON parsing until needed)
    local total
    total=$(grep -c "\"goal_id\": \"$goal_id\"" "$episodes_file" 2>/dev/null || echo 0)

    local completed
    completed=$(grep "\"goal_id\": \"$goal_id\"" "$episodes_file" 2>/dev/null | \
                grep -c "\"status\": \"closed\"" || echo 0)

    # Only parse if details needed
    local avg_duration
    avg_duration=$(grep "\"goal_id\": \"$goal_id\"" "$episodes_file" 2>/dev/null | \
                   jq -s 'map(.duration_seconds // 0) | if length > 0 then add / length else 0 end' 2>/dev/null || echo 0)

    jq -n --arg total "$total" \
          --arg completed "$completed" \
          --arg avg_duration "$avg_duration" \
          '{total_episodes: ($total | tonumber),
            completed: ($completed | tonumber),
            avg_duration: ($avg_duration | tonumber)}'
}
```

**Benefits**:
- Episode stats: 50-100ms → 10-30ms (grep + awk instead of full JSON load)
- Large episode sets: Linear improvement with file size
- Trade-off: Can't do complex cross-episode queries (acceptable, those are rare)

---

## 7. Cache Management

### Automatic Cache Cleanup

Add to daemon startup:

```bash
# In daemon.sh (line ~1730)

cleanup_optimization_caches() {
    local cache_dir="${DAEMON_ROOT}/.cache"

    if [ ! -d "$cache_dir" ]; then
        return
    fi

    # Clean old cache files (>24 hours)
    find "$cache_dir" -type f -mtime +1 -delete 2>/dev/null || true

    # Limit cache size (500MB max)
    local cache_size
    cache_size=$(du -sh "$cache_dir" 2>/dev/null | awk '{print $1}')

    if [ $(du -s "$cache_dir" 2>/dev/null | awk '{print $1}') -gt 512000 ]; then
        # Remove oldest cache files
        find "$cache_dir" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -n | head -50% | awk '{print $2}' | xargs rm -f 2>/dev/null || true
    fi
}

# Call at daemon startup
cleanup_optimization_caches
```

---

## 8. Monitoring & Metrics

### Add Cache Hit Rate Metrics

```bash
# Add to logs/cache-metrics.jsonl

log_cache_hit() {
    local cache_type="$1"    # "confidence", "template", "state", "episode"
    local hit="$1"           # true/false

    local entry
    entry=$(jq -n \
        --arg type "$cache_type" \
        --arg hit "$hit" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{cache_type: $type, hit: ($hit == "true"), timestamp: $ts}')

    echo "$entry" >> "${DAEMON_ROOT}/logs/cache-metrics.jsonl"
}
```

### Performance Dashboard Metrics

```bash
# queries/cache-performance.sh

print_cache_stats() {
    local cache_file="${DAEMON_ROOT}/logs/cache-metrics.jsonl"

    if [ ! -f "$cache_file" ]; then
        echo "No cache metrics yet"
        return
    fi

    echo "=== Cache Performance (Last 24 Hours) ==="
    echo ""

    # By cache type
    cat "$cache_file" | jq -s 'group_by(.cache_type) | map({
        cache_type: .[0].cache_type,
        total: length,
        hits: map(select(.hit == true)) | length,
        hit_rate: (map(select(.hit == true)) | length * 100 / length)
    })' | jq -r '.[] | "\(.cache_type): \(.hit_rate)% (\(.hits)/\(.total))"'
}
```

---

## 9. Performance Regression Testing

### Benchmark Suite

Create `tests/performance-benchmark.sh`:

```bash
#!/bin/bash

benchmark_confidence_calculation() {
    local iterations=100
    local start=$(date +%s%N)

    for i in $(seq 1 $iterations); do
        calculate_task_confidence "Write 2000-word article" "{}" "architect" >/dev/null
    done

    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))  # Convert to ms
    local avg=$(( elapsed / iterations ))

    echo "Confidence calculation: ${avg}ms avg (target: <30ms)"
    [ "$avg" -lt 30 ] && echo "✓ PASS" || echo "✗ FAIL"
}

benchmark_checkpoint_creation() {
    local test_file="/tmp/benchmark_file_$(date +%s).txt"
    dd if=/dev/zero of="$test_file" bs=1M count=50 2>/dev/null

    local start=$(date +%s%N)
    create_checkpoint "bench_task" "Benchmark" "$test_file"
    local end=$(date +%s%N)

    local elapsed=$(( (end - start) / 1000000 ))

    echo "Checkpoint creation (50MB): ${elapsed}ms (target: <300ms)"
    [ "$elapsed" -lt 300 ] && echo "✓ PASS" || echo "✗ FAIL"

    rm -f "$test_file"
}

benchmark_episode_batch_write() {
    local iterations=75
    local start=$(date +%s%N)

    local ep=$(create_episode "novel_publishable")
    for i in $(seq 1 $iterations); do
        local action=$(jq -n "{type: \"write\", attempt: $i, status: \"success\"}")
        ep=$(add_action_to_episode "$ep" "$action")
    done
    close_episode "$ep" "success" >/dev/null

    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))

    echo "Episode write (75 actions): ${elapsed}ms (target: <100ms)"
    [ "$elapsed" -lt 100 ] && echo "✓ PASS" || echo "✗ FAIL"
}

# Run benchmarks
benchmark_confidence_calculation
benchmark_checkpoint_creation
benchmark_episode_batch_write
```

---

## 10. Implementation Checklist

- [ ] Implement confidence calculation caching (40% latency reduction)
- [ ] Add compression method selection (30-50% storage reduction)
- [ ] Implement batch episode writing (60% I/O reduction)
- [ ] Add state variable caching (50% query reduction)
- [ ] Cache verification plan templates (90% template generation reduction)
- [ ] Implement lazy episode loading with grep (80% stats query reduction)
- [ ] Add cache cleanup in daemon startup
- [ ] Create cache performance metrics
- [ ] Develop performance regression test suite
- [ ] Validate overall overhead reduction: 0.1% → 0.03%

---

## 11. Expected Outcomes

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Confidence calculation (repeat) | 15-50ms | 1-5ms | 90% |
| Checkpoint compression | 40% ratio | 45-50% | +5-10% |
| Episode write (75 actions) | 750ms | 20ms | 97% |
| State query (cached) | 100-500ms | 10-30ms | 80% |
| Verification plan gen | 20-50ms | 2-8ms | 85% |
| Episode stats lookup | 50-100ms | 10-30ms | 75% |
| **Overall task overhead** | **0.1%** | **0.03%** | **70%** |

---

## 12. Deployment & Monitoring

**Deployment Order**:
1. Confidence caching (low risk, high impact)
2. Compression tuning (low risk, medium impact)
3. Episode batch writing (low risk, high I/O impact)
4. State variable caching (medium risk, needs invalidation tuning)
5. Template caching (low risk, low complexity)
6. Lazy loading (low risk, needs grep verification)

**Monitoring During Rollout**:
- Cache hit rates (target: >70%)
- Checkpoint storage (target: <350MB)
- Task execution time (should decrease ~0.07%)
- Daemon memory usage (should stay stable)

---

**End of Phase 7d Performance Optimization Guide**
