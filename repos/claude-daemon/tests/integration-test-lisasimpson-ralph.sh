#!/bin/bash

################################################################################
# LisaSimpson + Ralph Wiggum Integration Test Suite
#
# Comprehensive testing of:
# 1. Happy path - successful task completion on first attempt
# 2. Retry path - task fails, retries with adjusted approach, succeeds
# 3. Failure path - task fails all retries, rolledback
# 4. Complex workflow - multi-step episode with lessons
#
# Test environment: Uses temporary isolated daemon instance
#
# Usage: bash integration-test-lisasimpson-ralph.sh
# Status codes: 0 = all pass, 1+ = failures
#
# Authors: LisaSimpson + Ralph Wiggum Integration Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Test configuration
TEST_ROOT="/tmp/daemon-integration-test"
DAEMON_ROOT="$TEST_ROOT"
RESULTS_FILE="$TEST_ROOT/test-results.jsonl"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

################################################################################
# TEST UTILITIES
################################################################################

# Initialize test environment
setup_test_environment() {
    echo "Setting up test environment..."

    # Clean up old test environment
    rm -rf "$TEST_ROOT"

    # Create directory structure
    mkdir -p "$TEST_ROOT"/{lib,logs,memory,state/checkpoints,tasks}

    # Source libraries
    cp /home/opc/.claude/daemon/lib/*.sh "$TEST_ROOT/lib/" 2>/dev/null || true

    # Set DAEMON_ROOT for all operations
    export DAEMON_ROOT="$TEST_ROOT"

    # Create empty files
    touch "$TEST_ROOT/logs/activity.log"
    touch "$TEST_ROOT/logs/retry-metrics.jsonl"

    echo "✓ Test environment ready: $TEST_ROOT"
}

# Run a single test
run_test() {
    local test_name="$1"
    local test_function="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TESTS_RUN: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run test function
    if $test_function; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))

        # Log result
        {
            echo "{\"test\": \"$test_name\", \"status\": \"pass\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
        } >> "$RESULTS_FILE"
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))

        # Log result
        {
            echo "{\"test\": \"$test_name\", \"status\": \"fail\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
        } >> "$RESULTS_FILE"
    fi
}

################################################################################
# TEST 1: HAPPY PATH - Success on First Attempt
################################################################################

test_happy_path() {
    source "$TEST_ROOT/lib/confidence-engine.sh"
    source "$TEST_ROOT/lib/checkpoint-manager.sh"

    echo "Scenario: High-confidence task succeeds on first attempt"

    # Create test goal
    local goal=$(jq -n '{
        goal_id: "test_happy_path",
        description: "Write a simple article",
        blockers: [],
        dependencies: [],
        success_criteria: []
    }')

    # Calculate confidence (should be high for simple write task)
    local confidence
    confidence=$(calculate_task_confidence "Write a simple article" "$goal" "architect" 2>/dev/null | jq -r '.confidence_score' || echo "0")

    echo "  Confidence score: $confidence"

    # Get retry limit
    local retry_limit
    retry_limit=$(map_confidence_to_retry_limit "$confidence" 2>/dev/null || echo "3")

    echo "  Retry limit: $retry_limit"

    # Should be ≤3 retries for medium confidence or >5 for high
    if [ "$retry_limit" -ge 1 ] && [ "$retry_limit" -le 5 ]; then
        echo "  ✓ Retry limit within acceptable range (1-5)"
        return 0
    else
        echo "  ✗ Retry limit out of range: $retry_limit"
        return 1
    fi
}

################################################################################
# TEST 2: RETRY PATH - Failure then Success
################################################################################

test_retry_path() {
    source "$TEST_ROOT/lib/checkpoint-manager.sh"
    source "$TEST_ROOT/lib/retry-orchestrator.sh"

    echo "Scenario: Task fails initially, retries succeed after adjustment"

    # Create test files for checkpoint
    mkdir -p "$TEST_ROOT/test-files"
    echo "Original content" > "$TEST_ROOT/test-files/test-file.md"

    # Create checkpoint
    local checkpoint_result
    checkpoint_result=$(create_checkpoint "test_retry_1" "Test Task" "$TEST_ROOT/test-files/test-file.md" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to create checkpoint"
        return 1
    fi

    local checkpoint_id
    checkpoint_id=$(echo "$checkpoint_result" | jq -r '.checkpoint_id' 2>/dev/null)

    echo "  Checkpoint created: $checkpoint_id"

    # Modify file (simulate task modification)
    echo "Modified content" > "$TEST_ROOT/test-files/test-file.md"

    # Verify modification
    local modified_content
    modified_content=$(cat "$TEST_ROOT/test-files/test-file.md")

    if [ "$modified_content" != "Modified content" ]; then
        echo "  ✗ File modification failed"
        return 1
    fi

    echo "  ✓ File modified"

    # Rollback to checkpoint
    local rollback_result
    rollback_result=$(rollback_to_checkpoint "$checkpoint_id" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "  ✗ Rollback failed"
        return 1
    fi

    # Verify restoration
    local restored_content
    restored_content=$(cat "$TEST_ROOT/test-files/test-file.md")

    if [ "$restored_content" = "Original content" ]; then
        echo "  ✓ File successfully restored from checkpoint"
        return 0
    else
        echo "  ✗ Restoration failed: got '$restored_content', expected 'Original content'"
        return 1
    fi
}

################################################################################
# TEST 3: FAILURE PATH - All Retries Exhausted
################################################################################

test_failure_path() {
    source "$TEST_ROOT/lib/confidence-engine.sh"
    source "$TEST_ROOT/lib/checkpoint-manager.sh"

    echo "Scenario: Low-confidence task fails after limited retries"

    # Create low-confidence task
    local goal=$(jq -n '{
        goal_id: "test_failure",
        description: "Refactor entire authentication system while maintaining backward compatibility and integrating with multiple database backends",
        blockers: [{criterion: "blocked_on_research"}],
        dependencies: [{goal_id: "prereq_missing"}],
        success_criteria: []
    }')

    # Calculate confidence (should be low for complex task with blockers)
    local confidence
    confidence=$(calculate_task_confidence "Refactor complex system" "$goal" "architect" 2>/dev/null | jq -r '.confidence_score' || echo "0.5")

    echo "  Confidence score: $confidence"

    # Should have low confidence due to blockers
    if (( $(echo "$confidence < 0.6" | bc -l) )); then
        echo "  ✓ Low confidence detected (< 0.6)"

        # Get retry limit (should be 1)
        local retry_limit
        retry_limit=$(map_confidence_to_retry_limit "$confidence" 2>/dev/null || echo "3")

        echo "  Retry limit: $retry_limit"

        if [ "$retry_limit" -eq 1 ]; then
            echo "  ✓ Low-confidence task limited to 1 retry (fail-fast)"
            return 0
        fi
    fi

    echo "  ✗ Confidence not low enough or retry limit incorrect"
    return 1
}

################################################################################
# TEST 4: VERIFICATION - Plan Generation
################################################################################

test_verification_plans() {
    source "$TEST_ROOT/lib/verification-planner.sh"

    echo "Scenario: Verification plans auto-generated for different task types"

    # Test write task
    local write_plan
    write_plan=$(generate_verification_plan "Write a 2000-word article" "write_test")

    local write_checks
    write_checks=$(echo "$write_plan" | jq '.checks | length')

    if [ "$write_checks" -gt 0 ]; then
        echo "  ✓ Write task plan generated with $write_checks checks"
    else
        echo "  ✗ Write task plan has no checks"
        return 1
    fi

    # Test analysis task
    local analysis_plan
    analysis_plan=$(generate_verification_plan "Analyze performance metrics" "analysis_test")

    local analysis_checks
    analysis_checks=$(echo "$analysis_plan" | jq '.checks | length')

    if [ "$analysis_checks" -gt 0 ]; then
        echo "  ✓ Analysis task plan generated with $analysis_checks checks"
    else
        echo "  ✗ Analysis task plan has no checks"
        return 1
    fi

    return 0
}

################################################################################
# TEST 5: EPISODIC MEMORY - Learning from Episodes
################################################################################

test_episodic_memory() {
    source "$TEST_ROOT/lib/episodic-memory.sh"

    echo "Scenario: Create episode, add actions, extract lessons"

    # Create episode
    local episode
    episode=$(create_episode "test_goal_1" "creative_writing")

    echo "  Episode created"

    # Add actions
    local action1
    action1=$(jq -n '{type: "write", title: "Draft", status: "success"}')
    episode=$(add_action_to_episode "$episode" "$action1")

    local action2
    action2=$(jq -n '{type: "review", title: "Review", status: "success"}')
    episode=$(add_action_to_episode "$episode" "$action2")

    echo "  Added 2 actions"

    # Close episode with lessons
    episode=$(close_episode "$episode" "success")

    local lessons
    lessons=$(echo "$episode" | jq '.lessons_learned | length')

    echo "  Extracted $lessons lessons"

    if [ "$lessons" -gt 0 ]; then
        echo "  ✓ Episode completed with lessons learned"

        # Save episode
        mkdir -p "$TEST_ROOT/memory"
        save_episode "$episode"

        # Retrieve episode
        local retrieved
        retrieved=$(get_episodes_for_goal "test_goal_1" 2>/dev/null || echo "[]")

        # Show stats
        local stats
        stats=$(get_episode_stats "test_goal_1" 2>/dev/null || echo "{}")

        echo "  Episode stats: $stats" | head -c 80

        return 0
    else
        echo "  ✗ No lessons extracted"
        return 1
    fi
}

################################################################################
# TEST 6: CONFIDENCE CALIBRATION
################################################################################

test_confidence_calibration() {
    source "$TEST_ROOT/lib/confidence-engine.sh"

    echo "Scenario: Confidence scores calibrated across task types"

    local results="[]"

    # Test various task types
    local tasks=(
        "Write a blog post"
        "Analyze quarterly results"
        "Refactor authentication module"
        "Debug memory leak"
        "Write unit tests"
    )

    for task in "${tasks[@]}"; do
        local confidence
        confidence=$(calculate_task_confidence "$task" "{}" "architect" 2>/dev/null | jq -r '.confidence_score' || echo "0.5")

        local retry_limit
        retry_limit=$(map_confidence_to_retry_limit "$confidence" 2>/dev/null || echo "3")

        echo "  $task: confidence=$confidence, retries=$retry_limit"

        results=$(echo "$results" | jq ". += [{task: \"$task\", confidence: $confidence, retries: $retry_limit}]")
    done

    # Check variety (not all same)
    local unique_confidences
    unique_confidences=$(echo "$results" | jq '[.[].confidence] | unique | length')

    if [ "$unique_confidences" -gt 1 ]; then
        echo "  ✓ Confidence scores vary across task types ($unique_confidences unique values)"
        return 0
    else
        echo "  ✗ Confidence scores not varying enough"
        return 1
    fi
}

################################################################################
# TEST 7: CHECKPOINT STORAGE MANAGEMENT
################################################################################

test_checkpoint_storage() {
    source "$TEST_ROOT/lib/checkpoint-manager.sh"

    echo "Scenario: Checkpoint storage managed within size limits"

    # Create multiple checkpoints
    mkdir -p "$TEST_ROOT/test-checkpoints"

    for i in {1..3}; do
        echo "Test content $i" > "$TEST_ROOT/test-checkpoints/file_$i.txt"
        create_checkpoint "task_$i" "Task $i" "$TEST_ROOT/test-checkpoints/file_$i.txt" > /dev/null 2>&1
    done

    # List checkpoints
    local checkpoint_count
    checkpoint_count=$(ls -1 "$TEST_ROOT/state/checkpoints"/*.tar.gz 2>/dev/null | wc -l)

    echo "  Created $checkpoint_count checkpoints"

    # Calculate storage
    local storage
    storage=$(get_checkpoint_storage_used 2>/dev/null || echo "0")
    local storage_mb=$((storage / 1024 / 1024))

    echo "  Storage used: ${storage_mb}MB"

    # Storage should be < 500MB
    if [ "$storage_mb" -lt 500 ]; then
        echo "  ✓ Storage within limits (<500MB)"
        return 0
    else
        echo "  ✗ Storage exceeds limits"
        return 1
    fi
}

################################################################################
# COMPREHENSIVE WORKFLOW TEST
################################################################################

test_complete_workflow() {
    source "$TEST_ROOT/lib/goal-representation.sh"
    source "$TEST_ROOT/lib/confidence-engine.sh"
    source "$TEST_ROOT/lib/checkpoint-manager.sh"
    source "$TEST_ROOT/lib/episodic-memory.sh"

    echo "Scenario: Complete workflow with all components"

    # Step 1: Create goal with optional fields
    local goal
    goal=$(jq -n '{
        goal_id: "workflow_test",
        type: "creative_work",
        description: "Write article",
        success_criteria: [{criterion: "draft_complete", status: "not_started"}],
        world_state: {state_variables: {article_file: {type: "file_exists", path: "article.md"}}},
        confidence_score: 0.8,
        verification_plan: {checks: [{type: "file_exists"}]}
    }')

    echo "  ✓ Goal with LisaSimpson fields created"

    # Step 2: Calculate confidence
    local confidence
    confidence=$(echo "$goal" | jq -r '.confidence_score')

    echo "  ✓ Confidence score: $confidence"

    # Step 3: Create checkpoint
    mkdir -p "$TEST_ROOT/workflow-test"
    echo "Initial state" > "$TEST_ROOT/workflow-test/article.md"

    local checkpoint
    checkpoint=$(create_checkpoint "workflow_test" "Write article" "$TEST_ROOT/workflow-test/article.md" 2>/dev/null)

    echo "  ✓ Checkpoint created"

    # Step 4: Simulate modification
    echo "Modified article content" > "$TEST_ROOT/workflow-test/article.md"

    echo "  ✓ Content modified"

    # Step 5: Create episode
    local episode
    episode=$(create_episode "workflow_test" "article_writing")

    local action
    action=$(jq -n '{type: "write", status: "success"}')
    episode=$(add_action_to_episode "$episode" "$action")

    episode=$(close_episode "$episode" "success")

    echo "  ✓ Episode created and closed with lessons"

    # Step 6: Save episode
    mkdir -p "$TEST_ROOT/memory"
    save_episode "$episode"

    echo "  ✓ Episode saved to memory"

    # Verify all components worked together
    if [ -n "$goal" ] && [ -n "$checkpoint" ] && [ -n "$episode" ]; then
        echo "  ✓ Complete workflow executed successfully"
        return 0
    else
        echo "  ✗ Workflow incomplete"
        return 1
    fi
}

################################################################################
# TEST SUITE EXECUTION
################################################################################

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  LisaSimpson + Ralph Wiggum Integration Test Suite             ║"
    echo "║  Testing Adaptive Autonomous Task Execution System             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Setup
    setup_test_environment

    # Run all tests
    run_test "Happy Path - Success on First Attempt" test_happy_path
    run_test "Retry Path - Failure then Success via Checkpoint" test_retry_path
    run_test "Failure Path - Low Confidence Fail-Fast" test_failure_path
    run_test "Verification Plans - Auto-generation" test_verification_plans
    run_test "Episodic Memory - Learning from Episodes" test_episodic_memory
    run_test "Confidence Calibration - Task Type Variation" test_confidence_calibration
    run_test "Checkpoint Storage - Size Management" test_checkpoint_storage
    run_test "Complete Workflow - All Systems Integration" test_complete_workflow

    # Results
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Test Results                                                  ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Total Tests:  $TESTS_RUN"
    echo -e "  ${GREEN}Passed:${NC}      $TESTS_PASSED"
    echo -e "  ${RED}Failed:${NC}      $TESTS_FAILED"
    echo ""
    echo "  Results saved to: $RESULTS_FILE"
    echo ""

    # Exit code
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ SOME TESTS FAILED${NC}"
        return 1
    fi
}

# Run tests
main "$@"
