#!/bin/bash

################################################################################
# Verification Planner (LisaSimpson Integration)
#
# Automatically generates verification plans (OUTPUT/VERIFY criteria) from task
# descriptions. Enables self-referential verification where the system can
# check "Did I really complete this task?" after execution.
#
# Templates support:
# - Writing tasks (file creation, word count, content checks)
# - Analysis tasks (report generation, section presence)
# - Refactoring tasks (code structure, test pass/fail)
# - Testing tasks (coverage, test counts)
# - Research tasks (documentation, source citations)
#
# Authors: LisaSimpson + Autonomy Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# VERIFICATION PLAN STRUCTURE
################################################################################

# A verification plan contains checks that can be run after task completion:
# {
#   "task_id": "unique_task_id",
#   "checks": [
#     {
#       "type": "file_exists",
#       "description": "Output file must exist",
#       "parameters": {"path": "/path/to/file"}
#     },
#     {
#       "type": "word_count",
#       "description": "Minimum word count",
#       "parameters": {"path": "/path/to/file", "min": 2500}
#     }
#   ],
#   "success_criteria": "all_checks_pass",
#   "cleanup": ["optional", "files", "to", "clean", "up"]
# }

################################################################################
# TASK TYPE DETECTION
################################################################################

# Detect task type from description
# Returns: task_type (write, analyze, refactor, test, research, deploy)
detect_task_type() {
    local description="$1"

    local lower_desc
    lower_desc=$(echo "$description" | tr '[:upper:]' '[:lower:]')

    if echo "$lower_desc" | grep -qi "write\|create\|draft\|compose\|author\|chapter\|article"; then
        echo "write"
    elif echo "$lower_desc" | grep -qi "analyze\|analyze\|review\|evaluate\|assess\|examine"; then
        echo "analyze"
    elif echo "$lower_desc" | grep -qi "refactor\|optimize\|improve\|enhance\|rewrite"; then
        echo "refactor"
    elif echo "$lower_desc" | grep -qi "test\|verify\|validate\|check\|coverage"; then
        echo "test"
    elif echo "$lower_desc" | grep -qi "research\|investigate\|explore\|survey"; then
        echo "research"
    elif echo "$lower_desc" | grep -qi "deploy\|release\|publish\|launch\|build"; then
        echo "deploy"
    else
        echo "general"
    fi
}

# Extract expected output filenames from task description
# Returns: Array of filenames or file patterns
extract_output_files() {
    local description="$1"

    # Look for common output patterns
    local outputs='[]'

    # Match patterns like "write chapter-12.md", "create config.json", etc.
    if echo "$description" | grep -qi "chapter"; then
        # Extract chapter numbers if present
        if echo "$description" | grep -oE "chapter[- ]([0-9]+|[A-Z])" >/dev/null; then
            outputs=$(echo "$outputs" | jq '. += ["chapter-*.md"]')
        fi
    fi

    if echo "$description" | grep -qi "report\|document\|summary"; then
        outputs=$(echo "$outputs" | jq '. += ["*.md", "*.txt", "*.pdf"]' 2>/dev/null || echo "[]")
    fi

    if echo "$description" | grep -qi "config"; then
        outputs=$(echo "$outputs" | jq '. += ["*.json", "*.yaml", "*.yml"]')
    fi

    echo "$outputs"
}

################################################################################
# VERIFICATION PLAN TEMPLATES
################################################################################

# Generate verification plan for writing task
# Usage: plan_writing_task <task_description>
# Returns: Verification plan JSON
plan_writing_task() {
    local description="$1"

    # Extract expected word count if mentioned
    local min_words=1000
    if echo "$description" | grep -oE "[0-9]+\s*(word|words)" >/dev/null; then
        min_words=$(echo "$description" | grep -oE "[0-9]+" | head -1)
    fi

    # Check if specific file mentioned
    local output_file=""
    if echo "$description" | grep -oE "\.[a-z]+" >/dev/null; then
        output_file=$(echo "$description" | grep -oE "[^ ]*\.[a-z]{2,}" | head -1)
    fi

    if [ -z "$output_file" ]; then
        output_file="output.md"
    fi

    jq -n \
        --arg min_words "$min_words" \
        --arg output_file "$output_file" \
        '{
            task_type: "write",
            checks: [
                {
                    type: "file_exists",
                    description: "Output file must be created",
                    parameters: {path: $output_file}
                },
                {
                    type: "file_word_count",
                    description: ("Minimum " + $min_words + " words required"),
                    parameters: {path: $output_file, min: ($min_words | tonumber)}
                },
                {
                    type: "file_not_empty",
                    description: "File must contain content",
                    parameters: {path: $output_file}
                },
                {
                    type: "file_readable",
                    description: "File must be readable",
                    parameters: {path: $output_file}
                }
            ],
            success_criteria: "all_checks_pass"
        }'
}

# Generate verification plan for analysis task
# Usage: plan_analysis_task <task_description>
# Returns: Verification plan JSON
plan_analysis_task() {
    local description="$1"

    # Detect if specific structure expected
    local has_sections=false
    if echo "$description" | grep -qi "section\|chapter\|part\|module\|component"; then
        has_sections=true
    fi

    # Expect analysis output file
    local output_file="analysis.md"
    if echo "$description" | grep -oE "\.[a-z]+" >/dev/null; then
        output_file=$(echo "$description" | grep -oE "[^ ]*\.[a-z]{2,}" | head -1)
    fi

    local checks='[
        {
            "type": "file_exists",
            "description": "Analysis file must be created",
            "parameters": {"path": "'$output_file'"}
        },
        {
            "type": "file_line_count",
            "description": "Analysis should be substantial (minimum 50 lines)",
            "parameters": {"path": "'$output_file'", "min": 50}
        },
        {
            "type": "file_not_empty",
            "description": "File must contain analysis content",
            "parameters": {"path": "'$output_file'"}
        }
    ]'

    if [ "$has_sections" = true ]; then
        checks=$(echo "$checks" | jq \
            '. += [{
                "type": "file_contains",
                "description": "Analysis should have multiple sections",
                "parameters": {"path": "'$output_file'", "patterns": ["##", "###"]}
            }]')
    fi

    jq -n \
        --argjson checks "$checks" \
        '{
            task_type: "analyze",
            checks: $checks,
            success_criteria: "all_checks_pass"
        }'
}

# Generate verification plan for refactoring task
# Usage: plan_refactor_task <task_description>
# Returns: Verification plan JSON
plan_refactor_task() {
    local description="$1"

    jq -n '{
        task_type: "refactor",
        checks: [
            {
                type: "code_compiles",
                description: "Code must compile/have no syntax errors",
                parameters: {}
            },
            {
                type: "tests_pass",
                description: "All tests must pass after refactoring",
                parameters: {}
            },
            {
                type: "no_new_warnings",
                description: "No new compiler/linter warnings introduced",
                parameters: {}
            },
            {
                type: "functionality_preserved",
                description: "Refactored code must maintain original functionality",
                parameters: {}
            }
        ],
        success_criteria: "all_checks_pass"
    }'
}

# Generate verification plan for testing task
# Usage: plan_test_task <task_description>
# Returns: Verification plan JSON
plan_test_task() {
    local description="$1"

    # Look for coverage targets
    local min_coverage=80
    if echo "$description" | grep -oE "[0-9]+%\s*(coverage|covered)" >/dev/null; then
        min_coverage=$(echo "$description" | grep -oE "[0-9]+" | head -1)
    fi

    jq -n \
        --arg min_coverage "$min_coverage" \
        '{
            task_type: "test",
            checks: [
                {
                    type: "test_file_exists",
                    description: "Test file(s) must be created",
                    parameters: {pattern: "test_*.py"}
                },
                {
                    type: "tests_pass",
                    description: "All tests must pass",
                    parameters: {}
                },
                {
                    type: "coverage_threshold",
                    description: ("Minimum " + $min_coverage + "% code coverage"),
                    parameters: {min_percent: ($min_coverage | tonumber)}
                }
            ],
            success_criteria: "all_checks_pass"
        }'
}

# Generate verification plan for research task
# Usage: plan_research_task <task_description>
# Returns: Verification plan JSON
plan_research_task() {
    local description="$1"

    jq -n '{
        task_type: "research",
        checks: [
            {
                type: "findings_documented",
                description: "Research findings must be documented",
                parameters: {file_pattern: "*.md"}
            },
            {
                type: "sources_cited",
                description: "Sources must be cited",
                parameters: {required_formats: ["URL", "Author/Date", "Title"]}
            },
            {
                type: "analysis_present",
                description: "Must include analysis, not just raw data",
                parameters: {}
            }
        ],
        success_criteria: "all_checks_pass"
    }'
}

# Generate verification plan for deployment task
# Usage: plan_deploy_task <task_description>
# Returns: Verification plan JSON
plan_deploy_task() {
    local description="$1"

    jq -n '{
        task_type: "deploy",
        checks: [
            {
                type: "build_succeeds",
                description: "Build process must succeed",
                parameters: {}
            },
            {
                type: "tests_pass",
                description: "All tests must pass",
                parameters: {}
            },
            {
                type: "deployment_complete",
                description: "Deployment must be verified",
                parameters: {}
            },
            {
                type: "health_check",
                description: "System health check must pass",
                parameters: {}
            }
        ],
        success_criteria: "all_checks_pass"
    }'
}

################################################################################
# MAIN VERIFICATION PLAN GENERATION
################################################################################

# Generate complete verification plan from task description
# Usage: generate_verification_plan <task_description> [task_id]
# Returns: Verification plan JSON with auto-detected checks
generate_verification_plan() {
    local task_description="$1"
    local task_id="${2:-$(date +%s)}"

    # Detect task type
    local task_type
    task_type=$(detect_task_type "$task_description")

    # Generate appropriate plan
    local plan
    case "$task_type" in
        write)
            plan=$(plan_writing_task "$task_description")
            ;;
        analyze)
            plan=$(plan_analysis_task "$task_description")
            ;;
        refactor)
            plan=$(plan_refactor_task "$task_description")
            ;;
        test)
            plan=$(plan_test_task "$task_description")
            ;;
        research)
            plan=$(plan_research_task "$task_description")
            ;;
        deploy)
            plan=$(plan_deploy_task "$task_description")
            ;;
        *)
            # Generic plan
            plan=$(jq -n '{
                task_type: "general",
                checks: [
                    {
                        type: "task_completed",
                        description: "Task must be marked complete",
                        parameters: {}
                    }
                ],
                success_criteria: "all_checks_pass"
            }')
            ;;
    esac

    # Add task metadata
    plan=$(echo "$plan" | jq \
        --arg task_id "$task_id" \
        --arg description "$task_description" \
        '. + {task_id: $task_id, task_description: ($description | .[0:100]), timestamp: (now | floor | todate)}')

    echo "$plan"
}

################################################################################
# VERIFICATION PLAN VALIDATION
################################################################################

# Validate that a verification plan is well-formed
# Usage: validate_verification_plan <plan_json>
# Returns: JSON with validation result
validate_verification_plan() {
    local plan="$1"

    local is_valid=true
    local errors='[]'

    # Check required fields
    if ! echo "$plan" | jq -e '.checks | type == "array"' >/dev/null 2>&1; then
        is_valid=false
        errors=$(echo "$errors" | jq '. += ["Missing or invalid checks array"]')
    fi

    if ! echo "$plan" | jq -e '.success_criteria' >/dev/null 2>&1; then
        is_valid=false
        errors=$(echo "$errors" | jq '. += ["Missing success_criteria"]')
    fi

    # Check that at least one check exists
    local check_count
    check_count=$(echo "$plan" | jq '.checks | length')

    if [ "$check_count" -eq 0 ]; then
        is_valid=false
        errors=$(echo "$errors" | jq '. += ["Plan must have at least one check"]')
    fi

    jq -n \
        --argjson is_valid "$is_valid" \
        --argjson errors "$errors" \
        --argjson check_count "$check_count" \
        '{
            is_valid: $is_valid,
            errors: $errors,
            check_count: $check_count,
            validation_passed: ($is_valid and ($check_count > 0))
        }'
}

################################################################################
# EXPORTS
################################################################################

export -f detect_task_type
export -f extract_output_files
export -f plan_writing_task
export -f plan_analysis_task
export -f plan_refactor_task
export -f plan_test_task
export -f plan_research_task
export -f plan_deploy_task
export -f generate_verification_plan
export -f validate_verification_plan

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Verification Planner Self-Tests..." >&2
    echo ""

    # Test 1: Task type detection
    echo "Test 1: Task type detection" >&2
    echo "  'Write chapter 5' → $(detect_task_type "Write chapter 5")" >&2
    echo "  'Analyze performance' → $(detect_task_type "Analyze performance")" >&2
    echo "  'Refactor API layer' → $(detect_task_type "Refactor API layer")" >&2
    echo ""

    # Test 2: Write task verification plan
    echo "Test 2: Write task verification plan" >&2
    write_plan=$(generate_verification_plan "Write a 2500-word chapter about the protagonist")
    echo "$write_plan" | jq '{task_type, check_count: (.checks | length), checks: .checks[].type}' >&2
    echo ""

    # Test 3: Analysis task verification plan
    echo "Test 3: Analysis task verification plan" >&2
    analysis_plan=$(generate_verification_plan "Analyze the codebase and produce a report")
    echo "$analysis_plan" | jq '{task_type, check_count: (.checks | length)}' >&2
    echo ""

    # Test 4: Refactor task verification plan
    echo "Test 4: Refactor task verification plan" >&2
    refactor_plan=$(generate_verification_plan "Refactor the authentication module")
    echo "$refactor_plan" | jq '{task_type, checks: [.checks[].description]}' >&2
    echo ""

    # Test 5: Plan validation
    echo "Test 5: Verification plan validation" >&2
    write_plan=$(generate_verification_plan "Write documentation")
    validation=$(validate_verification_plan "$write_plan")
    echo "$validation" | jq '{is_valid, check_count, validation_passed}' >&2
    echo ""

    echo "✓ Verification Planner self-tests completed!" >&2
fi
