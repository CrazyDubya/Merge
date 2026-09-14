#!/bin/bash
# Benchmark jq subprocess overhead
#
# MAINTAINER NOTE: This benchmark validates the 73% performance improvement
# claimed by the batch read optimization. Run this before and after applying
# the optimization to verify the improvement.
#
# Usage: ./scripts/benchmark_jq.sh
# Expected output: ~14ms savings per 6-call sequence (87% reduction)

# Get daemon root directory
DAEMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# State file paths (using daemon root for safety)
STATE_FILE="${DAEMON_ROOT}/personalities/state.json"
EMOTIONAL_FILE="${DAEMON_ROOT}/triggers/emotional.json"

# Verify files exist before benchmarking
if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: State file not found: $STATE_FILE"
    exit 1
fi

if [ ! -f "$EMOTIONAL_FILE" ]; then
    echo "ERROR: Emotional state file not found: $EMOTIONAL_FILE"
    exit 1
fi

echo "=== JQ Subprocess Overhead Benchmark ==="
echo "Testing against: $EMOTIONAL_FILE"
echo ""

# Test 1: Single jq call baseline
echo "## Test 1: Single jq call (baseline):"
start=$(date +%s%N)
for i in {1..10}; do
    jq -r '.current_persona' "$STATE_FILE" > /dev/null
done
end=$(date +%s%N)
single_avg=$(( (end - start) / 10000000 ))
echo "  Average: ${single_avg}ms per call"
echo "  Interpretation: Fork+exec overhead for jq subprocess"
echo ""

# Test 2: Multiple sequential jq calls (current pattern)
echo "## Test 2: Sequential jq calls (check_emotional_triggers pattern):"
echo "  Simulating 6 sequential reads from same file"
start=$(date +%s%N)
for i in {1..10}; do
    jq -r '.current_state.frustration_level' "$EMOTIONAL_FILE" > /dev/null
    jq -r '.thresholds.high_frustration.value' "$EMOTIONAL_FILE" > /dev/null
    jq -r '.current_state.success_streak' "$EMOTIONAL_FILE" > /dev/null
    jq -r '.thresholds.success_streak_high.value' "$EMOTIONAL_FILE" > /dev/null
    jq -r '.current_state.failure_streak' "$EMOTIONAL_FILE" > /dev/null
    jq -r '.thresholds.failure_invoke_skeptic.value' "$EMOTIONAL_FILE" > /dev/null
done
end=$(date +%s%N)
sequential_avg=$(( (end - start) / 10000000 ))
echo "  Average: ${sequential_avg}ms per 6-call sequence"
echo "  Per-call: ~$((sequential_avg / 6))ms"
echo "  Overhead: 6 × fork+exec = 6 × subprocess creation"
echo ""

# Test 3: Single batched jq call (optimized pattern)
echo "## Test 3: Batched jq call (proposed optimization):"
echo "  Reading all 6 values in single jq invocation"
start=$(date +%s%N)
for i in {1..10}; do
    jq -r '{
        frustration: .current_state.frustration_level,
        frustration_thresh: .thresholds.high_frustration.value,
        success_streak: .current_state.success_streak,
        success_thresh: .thresholds.success_streak_high.value,
        failure_streak: .current_state.failure_streak,
        failure_thresh: .thresholds.failure_invoke_skeptic.value
    }' "$EMOTIONAL_FILE" > /dev/null
done
end=$(date +%s%N)
batched_avg=$(( (end - start) / 10000000 ))
echo "  Average: ${batched_avg}ms per batched call"
echo "  Overhead: 1 × fork+exec = 1 × subprocess creation"
echo ""

echo "## Performance Comparison:"
echo "  Sequential (6 calls): ${sequential_avg}ms"
echo "  Batched (1 call): ${batched_avg}ms"
savings=$(( sequential_avg - batched_avg ))
percent=$(( (savings * 100) / sequential_avg ))
echo "  Savings: ${savings}ms (${percent}% reduction)"
echo ""

echo "## Projected Impact on Daemon:"
echo "  Current: ~15 jq calls per wake cycle"
echo "  Optimized: ~3-4 jq calls per wake cycle (with batching)"
echo "  Estimated savings: ~$((savings * 2))ms per cycle"
echo "  Per-day savings (144 cycles): ~$((savings * 2 * 144 / 1000))s"
echo ""

# Validation
if [ "$percent" -lt 70 ]; then
    echo "⚠️  WARNING: Improvement is less than expected (${percent}% vs 87% expected)"
    echo "   This could indicate system load or different jq version"
else
    echo "✓ Performance improvement validated (${percent}% ≥ 70% threshold)"
fi

echo ""
echo "MAINTAINER NOTE: If improvement is <70%, investigate:"
echo "- System load during benchmark"
echo "- jq version differences"
echo "- File system caching effects"
echo "- Run benchmark multiple times for consistency"
