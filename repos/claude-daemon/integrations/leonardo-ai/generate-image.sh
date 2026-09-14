#!/usr/bin/env bash
# High-level image generation wrapper
# Makes it stupidly easy to generate images

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/leonardo-api.sh"
source "${SCRIPT_DIR}/lib/prompt-builder.sh"

IMAGES_DIR="${HOME}/.claude/daemon/generated-images"

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

generate_for_novel_chapter() {
    local chapter_number="$1"
    local scene_description="$2"
    local style="${3:-}"

    echo "📖 Generating illustration for Chapter $chapter_number"

    local prompt
    prompt=$(build_chapter_illustration_prompt "$scene_description" "$style")

    local settings
    settings=$(get_recommended_settings "$TEMPLATES_DIR/novel-prompts.json" "chapter_illustration")

    local output_dir="$IMAGES_DIR/novel/raw/chapter-${chapter_number}"

    echo "🎨 Prompt: $prompt"
    echo ""

    generate_and_download "$prompt" "$output_dir" "$settings"

    echo "✅ Chapter $chapter_number illustration saved to: $output_dir"
}

generate_toaster_character() {
    local context="$1"
    local personality="${2:-}"
    local output_name="${3:-toaster-character}"

    echo "🍞 Generating sentient toaster character"

    local prompt
    prompt=$(build_toaster_prompt "$context" "$personality")

    local settings
    settings=$(get_recommended_settings "$TEMPLATES_DIR/novel-prompts.json" "toaster_specific")

    local output_dir="$IMAGES_DIR/novel/raw/$output_name"

    echo "🎨 Prompt: $prompt"
    echo ""

    generate_and_download "$prompt" "$output_dir" "$settings"

    echo "✅ Toaster character saved to: $output_dir"
}

generate_persona_avatar() {
    local persona_name="$1"

    echo "🎭 Generating avatar for: $persona_name"

    local prompt
    prompt=$(build_persona_avatar_prompt "$persona_name")

    local settings
    settings=$(get_recommended_settings "$TEMPLATES_DIR/dashboard-prompts.json" "persona_avatars")

    local output_dir="$IMAGES_DIR/dashboard/raw/avatars"

    echo "🎨 Prompt: $prompt"
    echo ""

    generate_and_download "$prompt" "$output_dir" "$settings"

    echo "✅ Avatar for $persona_name saved to: $output_dir"
}

generate_dashboard_icon() {
    local icon_subject="$1"
    local style="${2:-}"
    local output_name="${3:-icon}"

    echo "🎨 Generating dashboard icon: $icon_subject"

    local prompt
    prompt=$(build_icon_prompt "$icon_subject" "$style")

    local settings
    settings=$(get_recommended_settings "$TEMPLATES_DIR/dashboard-prompts.json" "icons_and_ui")

    local output_dir="$IMAGES_DIR/dashboard/raw/icons"

    echo "🎨 Prompt: $prompt"
    echo ""

    generate_and_download "$prompt" "$output_dir" "$settings"

    echo "✅ Icon saved to: $output_dir"
}

generate_custom() {
    local prompt="$1"
    local output_dir="$2"
    local options="${3:-{}}"

    echo "🎨 Custom generation"
    echo "📝 Prompt: $prompt"
    echo "📁 Output: $output_dir"
    echo ""

    generate_and_download "$prompt" "$output_dir" "$options"

    echo "✅ Image saved to: $output_dir"
}

# =============================================================================
# BATCH OPERATIONS
# =============================================================================

generate_all_persona_avatars() {
    local personas=("architect" "optimizer" "auditor" "maintainer" "skeptic" "experimenter")

    echo "🎭 Generating avatars for all personas..."
    echo ""

    for persona in "${personas[@]}"; do
        generate_persona_avatar "$persona"
        echo ""
        sleep 2  # Be nice to the API
    done

    echo "✅ All persona avatars generated!"
}

generate_toaster_character_sheet() {
    echo "🍞 Generating complete toaster character sheet..."
    echo ""

    local contexts=(
        "contemplating existence in kitchen:philosophical"
        "preparing for adventure:confident"
        "confronting antagonist:fierce"
        "moment of triumph:triumphant"
        "experiencing loss:melancholic"
    )

    local count=1
    for context_pair in "${contexts[@]}"; do
        local context="${context_pair%:*}"
        local personality="${context_pair#*:}"

        generate_toaster_character "$context" "$personality" "toaster-expression-${count}"
        echo ""
        sleep 2
        ((count++))
    done

    echo "✅ Complete toaster character sheet generated!"
}

# =============================================================================
# TESTING/DEMO
# =============================================================================

test_integration() {
    echo "🧪 Testing Leonardo.ai integration..."
    echo ""

    # Test 1: Check if API key is configured
    echo "Test 1: API key configuration"
    if get_api_key &>/dev/null; then
        echo "✅ API key found"
    else
        echo "❌ API key not configured"
        echo ""
        echo "To configure:"
        echo "  export LEONARDO_API_KEY='your-key-here'"
        echo "  OR"
        echo "  echo 'API_KEY=your-key-here' > ~/.claude/daemon/integrations/leonardo-ai/.config"
        return 1
    fi
    echo ""

    # Test 2: Prompt generation
    echo "Test 2: Prompt generation"
    local test_prompt
    test_prompt=$(build_toaster_prompt "testing the system" "cheerful")
    echo "Generated prompt: $test_prompt"
    echo "✅ Prompt builder working"
    echo ""

    # Test 3: Template loading
    echo "Test 3: Template settings"
    local test_settings
    test_settings=$(get_recommended_settings "$TEMPLATES_DIR/novel-prompts.json" "toaster_specific")
    echo "Loaded settings: $test_settings"
    echo "✅ Template loader working"
    echo ""

    # Test 4: API connectivity (optional - only if --full flag)
    if [[ "${1:-}" == "--full" ]]; then
        echo "Test 4: API connectivity"
        if get_user_info &>/dev/null; then
            echo "✅ API connection successful"
        else
            echo "⚠️  API connection failed (check key validity)"
        fi
        echo ""
    fi

    echo "✅ Integration tests passed!"
}

# =============================================================================
# CLI INTERFACE
# =============================================================================

show_usage() {
    cat << EOF
Leonardo.ai Image Generation - Easy High-Level Interface

NOVEL COMMANDS:
  chapter <number> <description> [style]
    Generate chapter illustration
    Example: $0 chapter 1 "toaster standing in rain-soaked alley"

  toaster <context> [personality] [output_name]
    Generate toaster character image
    Example: $0 toaster "experiencing first sunrise" "philosophical"

  character-sheet
    Generate complete set of toaster expressions (5 images)

DASHBOARD COMMANDS:
  avatar <persona_name>
    Generate avatar for specific persona
    Example: $0 avatar experimenter

  avatars-all
    Generate avatars for all 6 personas

  icon <subject> [style] [output_name]
    Generate dashboard icon
    Example: $0 icon "task completion badge"

CUSTOM:
  custom <prompt> <output_dir> [options_json]
    Generate with custom prompt and settings
    Example: $0 custom "amazing image" ./my-images '{"width":1024,"height":1024}'

UTILITIES:
  test [--full]
    Test integration setup (use --full to test API connection)

  prompt <type> <args...>
    Generate prompt without creating image (for testing)
    Example: $0 prompt toaster "contemplating toast"

EXAMPLES:
  # Generate chapter 1 illustration
  $0 chapter 1 "a chrome toaster standing alone in a futuristic kitchen"

  # Generate toaster character portrait
  $0 toaster "the moment of awakening" "confused and curious"

  # Generate all persona avatars
  $0 avatars-all

  # Test setup
  $0 test

EOF
}

# Main command router
case "${1:-}" in
    chapter)
        generate_for_novel_chapter "${2}" "${3}" "${4:-}"
        ;;
    toaster)
        generate_toaster_character "${2}" "${3:-}" "${4:-toaster-character}"
        ;;
    character-sheet)
        generate_toaster_character_sheet
        ;;
    avatar)
        generate_persona_avatar "${2}"
        ;;
    avatars-all)
        generate_all_persona_avatars
        ;;
    icon)
        generate_dashboard_icon "${2}" "${3:-}" "${4:-icon}"
        ;;
    custom)
        generate_custom "${2}" "${3}" "${4:-{}}"
        ;;
    test)
        test_integration "${2:-}"
        ;;
    prompt)
        # Just generate prompt, don't create image
        shift
        "${SCRIPT_DIR}/lib/prompt-builder.sh" "$@"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        show_usage
        ;;
esac
