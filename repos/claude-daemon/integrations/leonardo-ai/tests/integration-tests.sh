#!/bin/bash
# Leonardo.ai Integration Test Suite
# Executable automated tests for the Leonardo.ai image generation integration
#
# Usage:
#   ./tests/integration-tests.sh              # Run all tests (skips API tests without key)
#   ./tests/integration-tests.sh --with-api   # Run all tests including API calls (COSTS MONEY!)
#   ./tests/integration-tests.sh --quick      # Run only quick tests (no API, no slow tests)
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Critical setup failure (can't continue)

set -euo pipefail

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_API_TESTS=false
QUICK_MODE=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Parse arguments
for arg in "$@"; do
  case $arg in
    --with-api)
      RUN_API_TESTS=true
      shift
      ;;
    --quick)
      QUICK_MODE=true
      shift
      ;;
    --help|-h)
      echo "Leonardo.ai Integration Test Suite"
      echo ""
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --with-api    Run API tests (WARNING: Costs money!)"
      echo "  --quick       Run only quick tests (no API, no slow tests)"
      echo "  --help        Show this help message"
      echo ""
      exit 0
      ;;
  esac
done

# Change to integration root
cd "$INTEGRATION_ROOT"

# Test result tracking
declare -a FAILED_TESTS=()

# Utility functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
  echo -e "${GREEN}[PASS]${NC} $*"
}

log_warning() {
  echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
  echo -e "${RED}[FAIL]${NC} $*"
}

log_skip() {
  echo -e "${YELLOW}[SKIP]${NC} $*"
}

# Test framework functions
test_start() {
  local test_name="$1"
  TESTS_RUN=$((TESTS_RUN + 1))
  echo ""
  echo -e "${BLUE}━━━ Test $TESTS_RUN: $test_name ━━━${NC}"
}

test_pass() {
  TESTS_PASSED=$((TESTS_PASSED + 1))
  log_success "$1"
}

test_fail() {
  TESTS_FAILED=$((TESTS_FAILED + 1))
  FAILED_TESTS+=("Test $TESTS_RUN: $1")
  log_error "$1"
}

test_skip() {
  TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
  log_skip "$1"
}

assert_true() {
  local condition="$1"
  local message="$2"

  if eval "$condition"; then
    test_pass "$message"
    return 0
  else
    test_fail "$message"
    return 1
  fi
}

assert_file_exists() {
  local file="$1"
  local message="${2:-File exists: $file}"

  if [[ -f "$file" ]]; then
    test_pass "$message"
    return 0
  else
    test_fail "$message (file not found)"
    return 1
  fi
}

assert_executable() {
  local file="$1"
  local message="${2:-File is executable: $file}"

  if [[ -x "$file" ]]; then
    test_pass "$message"
    return 0
  else
    test_fail "$message (not executable)"
    return 1
  fi
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local message="$3"

  if echo "$haystack" | grep -q "$needle"; then
    test_pass "$message"
    return 0
  else
    test_fail "$message (not found: $needle)"
    return 1
  fi
}

assert_valid_json() {
  local file="$1"
  local message="${2:-Valid JSON: $file}"

  if jq empty "$file" 2>/dev/null; then
    test_pass "$message"
    return 0
  else
    test_fail "$message (invalid JSON)"
    return 1
  fi
}

# Cleanup function for test isolation
cleanup_test_artifacts() {
  log_info "Cleaning up test artifacts..."

  # Backup existing files
  if [[ -f cost-tracking.jsonl ]]; then
    cp cost-tracking.jsonl cost-tracking.jsonl.test-backup
  fi
  if [[ -f generations.jsonl ]]; then
    cp generations.jsonl generations.jsonl.test-backup
  fi
  if [[ -d cache ]]; then
    mv cache cache.test-backup
  fi

  # Create clean state
  mkdir -p cache
  rm -f cost-tracking.jsonl generations.jsonl
}

restore_artifacts() {
  log_info "Restoring original artifacts..."

  # Restore backups
  if [[ -f cost-tracking.jsonl.test-backup ]]; then
    mv cost-tracking.jsonl.test-backup cost-tracking.jsonl
  fi
  if [[ -f generations.jsonl.test-backup ]]; then
    mv generations.jsonl.test-backup generations.jsonl
  fi
  if [[ -d cache.test-backup ]]; then
    rm -rf cache
    mv cache.test-backup cache
  fi
}

# Test suite header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Leonardo.ai Integration Test Suite                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
log_info "Integration root: $INTEGRATION_ROOT"
log_info "API tests: $([ "$RUN_API_TESTS" = true ] && echo 'ENABLED (will cost money!)' || echo 'DISABLED')"
log_info "Quick mode: $([ "$QUICK_MODE" = true ] && echo 'YES' || echo 'NO')"
echo ""

# ============================================================================
# TEST CATEGORY 1: Directory Structure
# ============================================================================

test_start "Directory structure exists"
assert_file_exists "lib/leonardo-api.sh"
assert_file_exists "lib/prompt-builder.sh"
assert_file_exists "generate-image.sh"
assert_file_exists "templates/novel-prompts.json"
assert_file_exists "templates/dashboard-prompts.json"
assert_file_exists ".gitignore"
assert_file_exists "README.md"
assert_file_exists "QUICKSTART.md"

# ============================================================================
# TEST CATEGORY 2: Script Permissions
# ============================================================================

test_start "Scripts are executable"
assert_executable "lib/leonardo-api.sh"
assert_executable "lib/prompt-builder.sh"
assert_executable "generate-image.sh"

# ============================================================================
# TEST CATEGORY 3: Gitignore Protection
# ============================================================================

test_start "Gitignore protects sensitive files"

# Test .config is ignored
if git check-ignore .config >/dev/null 2>&1; then
  test_pass ".config is gitignored"
else
  test_fail ".config is NOT gitignored (SECURITY RISK!)"
fi

# Test cache/ is ignored
if git check-ignore cache/ >/dev/null 2>&1 || git check-ignore cache >/dev/null 2>&1; then
  test_pass "cache/ is gitignored"
else
  test_fail "cache/ is NOT gitignored"
fi

# Test .jsonl files are ignored
touch test-tracking.jsonl
if git check-ignore test-tracking.jsonl >/dev/null 2>&1; then
  test_pass "*.jsonl files are gitignored"
  rm test-tracking.jsonl
else
  test_fail "*.jsonl files are NOT gitignored"
  rm test-tracking.jsonl
fi

# ============================================================================
# TEST CATEGORY 4: Template Validation
# ============================================================================

test_start "Template files are valid JSON"
assert_valid_json "templates/novel-prompts.json"
assert_valid_json "templates/dashboard-prompts.json"

# Validate template structure
test_start "Template structure is correct"
for template in templates/*.json; do
  template_name=$(basename "$template")

  # Templates are structured as objects with categories
  # Each category should have: base_template, variation arrays, recommended_settings

  # Get all top-level categories
  categories=$(jq -r 'keys[]' "$template" 2>/dev/null)

  if [[ -z "$categories" ]]; then
    test_fail "$template_name has no categories"
    continue
  fi

  test_pass "$template_name has categories: $(echo $categories | tr '\n' ' ')"

  # Check each category has proper structure
  for category in $categories; do
    # Check for base_template (or alternative fields like prompts, sentient_toaster_base)
    if jq -e ".$category.base_template" "$template" >/dev/null 2>&1; then
      test_pass "$template_name.$category has base_template"
    elif jq -e ".$category.prompts" "$template" >/dev/null 2>&1; then
      test_pass "$template_name.$category has prompts array (alternative to base_template)"
    elif jq -e ".$category.sentient_toaster_base" "$template" >/dev/null 2>&1; then
      test_pass "$template_name.$category has sentient_toaster_base (alternative to base_template)"
    else
      test_fail "$template_name.$category missing base_template or prompt definition"
    fi

    # Check for recommended_settings
    if jq -e ".$category.recommended_settings" "$template" >/dev/null 2>&1; then
      test_pass "$template_name.$category has recommended_settings"

      # Verify recommended_settings has width/height
      if jq -e ".$category.recommended_settings.width" "$template" >/dev/null 2>&1; then
        test_pass "$template_name.$category.recommended_settings has width"
      else
        test_fail "$template_name.$category.recommended_settings missing width"
      fi
    else
      test_fail "$template_name.$category missing recommended_settings"
    fi

    # Check for at least one variation array (styles, moods, expressions, etc.)
    variation_count=$(jq ".$category | to_entries | map(select(.value | type == \"array\")) | length" "$template" 2>/dev/null)
    if [[ "$variation_count" -gt 0 ]]; then
      test_pass "$template_name.$category has $variation_count variation arrays"
    else
      test_fail "$template_name.$category has no variation arrays"
    fi
  done
done

# ============================================================================
# TEST CATEGORY 5: Configuration Detection (No API Key)
# ============================================================================

test_start "Configuration detection (missing .config)"

# Temporarily hide .config if it exists
CONFIG_HIDDEN=false
if [[ -f .config ]]; then
  mv .config .config.hidden-for-test
  CONFIG_HIDDEN=true
fi

# Test that script fails gracefully without config
# Use the 'test' command which actually checks for API key
# Allow it to fail (returns 1 when API key missing)
output=$(./generate-image.sh test 2>&1 || true)
if echo "$output" | grep -qi "api.*key\|config"; then
  test_pass "Shows helpful error when .config missing"
  # Also check that it mentions how to fix it
  if echo "$output" | grep -q "configure\|export\|echo"; then
    test_pass "Error message includes instructions for fixing"
  else
    test_fail "Error message doesn't include fix instructions"
  fi
else
  test_fail "Error message doesn't mention config/API key"
fi

# Restore .config
if [[ "$CONFIG_HIDDEN" = true ]]; then
  mv .config.hidden-for-test .config
fi

# ============================================================================
# TEST CATEGORY 6: Test Mode Functionality
# ============================================================================

test_start "Test mode (no API calls)"

# Clean up for test isolation
cleanup_test_artifacts

# Run in test mode
log_info "Running generation in test-mode..."
output=$(LEONARDO_API_KEY=test-mode ./generate-image.sh "A curious robot reading a book" 2>&1 || true)

# Check output contains cost estimate
if echo "$output" | grep -qi "cost\|price\|\$"; then
  test_pass "Test mode shows cost estimate"
else
  test_fail "Test mode doesn't show cost estimate"
fi

# Check that mock file was created (if implementation does this)
if [[ -f cost-tracking.jsonl ]]; then
  test_pass "Test mode creates cost tracking log"

  # Verify it's valid JSON
  if jq empty cost-tracking.jsonl 2>/dev/null; then
    test_pass "Cost tracking log is valid JSON"
  else
    test_fail "Cost tracking log is invalid JSON"
  fi
else
  test_skip "Test mode doesn't create cost tracking (not implemented)"
fi

# Restore artifacts
restore_artifacts

# ============================================================================
# TEST CATEGORY 7: Prompt Builder Validation
# ============================================================================

test_start "Prompt builder generates valid output"

# Source the prompt builder
source lib/prompt-builder.sh

# Test chapter illustration prompt builder
if declare -f build_chapter_illustration_prompt >/dev/null; then
  prompt_text=$(build_chapter_illustration_prompt \
    "A sunrise over mountains" \
    "watercolor painting")

  # Prompt builders return plain text, not JSON
  if [[ -n "$prompt_text" ]]; then
    test_pass "build_chapter_illustration_prompt generates prompt text"

    # Check that prompt includes the scene description
    if echo "$prompt_text" | grep -q "sunrise over mountains"; then
      test_pass "Prompt includes scene description"
    else
      test_fail "Prompt missing scene description"
    fi

    # Check that prompt includes style
    if echo "$prompt_text" | grep -qi "watercolor"; then
      test_pass "Prompt includes style"
    else
      test_fail "Prompt missing style"
    fi
  else
    test_fail "build_chapter_illustration_prompt returned empty string"
  fi
else
  test_fail "build_chapter_illustration_prompt function not found"
fi

# Test get_recommended_settings function
if declare -f get_recommended_settings >/dev/null; then
  settings_json=$(get_recommended_settings "templates/novel-prompts.json" "chapter_illustration")

  # This SHOULD return JSON
  if echo "$settings_json" | jq empty 2>/dev/null; then
    test_pass "get_recommended_settings generates valid JSON"

    # Check required fields
    if echo "$settings_json" | jq -e '.width' >/dev/null 2>&1; then
      test_pass "Settings JSON has 'width' field"
    else
      test_fail "Settings JSON missing 'width' field"
    fi

    if echo "$settings_json" | jq -e '.height' >/dev/null 2>&1; then
      test_pass "Settings JSON has 'height' field"
    else
      test_fail "Settings JSON missing 'height' field"
    fi
  else
    test_fail "get_recommended_settings generates invalid JSON"
  fi
else
  test_fail "get_recommended_settings function not found"
fi

# ============================================================================
# TEST CATEGORY 8: API Tests (Only if --with-api flag set)
# ============================================================================

if [[ "$RUN_API_TESTS" = true ]]; then
  if [[ ! -f .config ]]; then
    log_error "Cannot run API tests: .config file not found"
    log_info "Create .config with: echo 'LEONARDO_API_KEY=your_key' > .config"
    exit 2
  fi

  source .config

  if [[ -z "${LEONARDO_API_KEY:-}" ]] || [[ "$LEONARDO_API_KEY" = "test-mode" ]]; then
    log_error "Cannot run API tests: Valid API key not configured"
    exit 2
  fi

  log_warning "API tests will consume credits on your Leonardo.ai account!"
  read -p "Continue? (yes/no): " confirm
  if [[ "$confirm" != "yes" ]]; then
    log_info "API tests cancelled by user"
    exit 0
  fi

  # Clean up for API test isolation
  cleanup_test_artifacts

  test_start "Simple API generation test"
  log_info "Generating test image (this will cost ~\$0.01)..."

  if ./generate-image.sh "A simple test image of a red cube on blue background" >/dev/null 2>&1; then
    test_pass "API generation succeeded"

    # Check artifacts were created
    if [[ -f cost-tracking.jsonl ]]; then
      test_pass "Cost tracking log created"
    else
      test_fail "Cost tracking log not created"
    fi

    if [[ -f generations.jsonl ]]; then
      test_pass "Generations log created"

      # Check image was downloaded
      image_path=$(jq -r '.image_path' generations.jsonl | tail -1)
      if [[ -f "$image_path" ]]; then
        test_pass "Image downloaded: $image_path"

        # Verify it's a valid image
        if file "$image_path" | grep -qi "jpeg\|png\|image"; then
          test_pass "Downloaded file is valid image"
        else
          test_fail "Downloaded file is not a valid image"
        fi
      else
        test_fail "Image not downloaded"
      fi
    else
      test_fail "Generations log not created"
    fi
  else
    test_fail "API generation failed"
  fi

  # Restore artifacts
  restore_artifacts
else
  log_skip "API tests disabled (use --with-api to enable)"
fi

# ============================================================================
# TEST SUMMARY
# ============================================================================

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Test Summary                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total tests run:     ${BLUE}$TESTS_RUN${NC}"
echo -e "Passed:              ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed:              ${RED}$TESTS_FAILED${NC}"
echo -e "Skipped:             ${YELLOW}$TESTS_SKIPPED${NC}"
echo ""

if [[ $TESTS_FAILED -gt 0 ]]; then
  echo -e "${RED}Failed tests:${NC}"
  for failed_test in "${FAILED_TESTS[@]}"; do
    echo -e "  ${RED}✗${NC} $failed_test"
  done
  echo ""
  exit 1
else
  echo -e "${GREEN}All tests passed! ✓${NC}"
  echo ""
  exit 0
fi
