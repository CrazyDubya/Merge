#!/usr/bin/env bash
# Prompt Builder - Smart prompt generation from templates
# Makes it easy to create great prompts without overthinking

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="${SCRIPT_DIR}/../templates"

# =============================================================================
# TEMPLATE LOADING
# =============================================================================

load_template() {
    local template_file="$1"
    local category="$2"

    if [[ ! -f "$template_file" ]]; then
        echo "ERROR: Template file not found: $template_file" >&2
        return 1
    fi

    jq -r ".${category}" "$template_file"
}

# =============================================================================
# RANDOM SELECTION HELPERS
# =============================================================================

random_from_array() {
    local json_array="$1"
    echo "$json_array" | jq -r '.[]' | shuf -n 1
}

random_value() {
    local template="$1"
    local field="$2"

    echo "$template" | jq -r ".${field}[]" | shuf -n 1
}

# =============================================================================
# NOVEL PROMPT BUILDERS
# =============================================================================

build_chapter_illustration_prompt() {
    local scene_description="$1"
    local custom_style="${2:-}"

    local template
    template=$(load_template "$TEMPLATES_DIR/novel-prompts.json" "chapter_illustration")

    local base=$(echo "$template" | jq -r '.base_template')
    local style=${custom_style:-$(random_value "$template" "styles")}
    local mood=$(random_value "$template" "moods")
    local lighting=$(random_value "$template" "lighting")

    # Replace placeholders
    local prompt="$base"
    prompt="${prompt//\{scene_description\}/$scene_description}"
    prompt="${prompt//\{style\}/$style}"
    prompt="${prompt//\{mood\}/$mood}"
    prompt="${prompt//\{lighting\}/$lighting}"

    echo "$prompt"
}

build_character_portrait_prompt() {
    local character_description="$1"
    local expression="${2:-}"
    local custom_style="${3:-}"

    local template
    template=$(load_template "$TEMPLATES_DIR/novel-prompts.json" "character_portrait")

    local base=$(echo "$template" | jq -r '.base_template')
    local expr=${expression:-$(random_value "$template" "expressions")}
    local style=${custom_style:-$(random_value "$template" "art_styles")}
    local clothing="professional attire"  # Could be parameterized
    local background=$(random_value "$template" "backgrounds")

    local prompt="$base"
    prompt="${prompt//\{character_description\}/$character_description}"
    prompt="${prompt//\{expression\}/$expr}"
    prompt="${prompt//\{clothing_style\}/$clothing}"
    prompt="${prompt//\{art_style\}/$style}"
    prompt="${prompt//\{background\}/$background}"

    echo "$prompt"
}

build_scene_visualization_prompt() {
    local focal_point="$1"
    local custom_location="${2:-}"

    local template
    template=$(load_template "$TEMPLATES_DIR/novel-prompts.json" "scene_visualization")

    local base=$(echo "$template" | jq -r '.base_template')
    local location=${custom_location:-$(random_value "$template" "locations")}
    local time=$(random_value "$template" "times_of_day")
    local weather=$(random_value "$template" "times_of_day")  # Reuse for atmosphere
    local style=$(random_value "$template" "artistic_styles")
    local detail=$(random_value "$template" "detail_levels")

    local prompt="$base"
    prompt="${prompt//\{location\}/$location}"
    prompt="${prompt//\{time_of_day\}/$time}"
    prompt="${prompt//\{weather_atmosphere\}/$weather}"
    prompt="${prompt//\{focal_point\}/$focal_point}"
    prompt="${prompt//\{artistic_style\}/$style}"
    prompt="${prompt//\{detail_level\}/$detail}"

    echo "$prompt"
}

build_toaster_prompt() {
    local context="$1"
    local personality="${2:-}"

    local template
    template=$(load_template "$TEMPLATES_DIR/novel-prompts.json" "toaster_specific")

    local base=$(echo "$template" | jq -r '.sentient_toaster_base')
    local traits=${personality:-$(random_value "$template" "personality_traits")}
    local physical=$(random_value "$template" "physical_descriptions")
    local style="detailed character illustration"
    local emotion=$(random_value "$template" "emotion_indicators")

    local prompt="$base"
    prompt="${prompt//\{personality_traits\}/$traits}"
    prompt="${prompt//\{physical_description\}/$physical}"
    prompt="${prompt//\{setting\}/$context}"
    prompt="${prompt//\{art_style\}/$style}"
    prompt="${prompt//\{emotion_indicators\}/$emotion}"

    echo "$prompt"
}

# =============================================================================
# DASHBOARD PROMPT BUILDERS
# =============================================================================

build_persona_avatar_prompt() {
    local persona_name="$1"

    local template
    template=$(load_template "$TEMPLATES_DIR/dashboard-prompts.json" "persona_avatars")

    local persona_data
    persona_data=$(echo "$template" | jq ".personas.${persona_name}")

    if [[ "$persona_data" == "null" ]]; then
        echo "ERROR: Unknown persona: $persona_name" >&2
        return 1
    fi

    local base=$(echo "$template" | jq -r '.base_template')
    local metaphor=$(echo "$persona_data" | jq -r '.visual_metaphors[]' | shuf -n 1)
    local mood=$(echo "$persona_data" | jq -r '.mood')
    local style=$(random_value "$template" "art_styles")
    local composition=$(random_value "$template" "compositions")

    local prompt="$base"
    prompt="${prompt//\{persona_name\}/$persona_name}"
    prompt="${prompt//\{visual_metaphor\}/$metaphor}"
    prompt="${prompt//\{art_style\}/$style}"
    prompt="${prompt//\{mood\}/$mood}"
    prompt="${prompt//\{composition\}/$composition}"

    echo "$prompt"
}

build_icon_prompt() {
    local icon_subject="$1"
    local custom_style="${2:-}"

    local template
    template=$(load_template "$TEMPLATES_DIR/dashboard-prompts.json" "icons_and_ui")

    local base=$(echo "$template" | jq -r '.base_template')
    local style=${custom_style:-$(random_value "$template" "styles")}
    local colors=$(random_value "$template" "color_schemes")
    local size_note="suitable for dashboard UI, scalable"

    local prompt="$base"
    prompt="${prompt//\{icon_subject\}/$icon_subject}"
    prompt="${prompt//\{style\}/$style}"
    prompt="${prompt//\{color_scheme\}/$colors}"
    prompt="${prompt//\{size_note\}/$size_note}"

    echo "$prompt"
}

# =============================================================================
# SETTINGS HELPERS
# =============================================================================

get_recommended_settings() {
    local template_file="$1"
    local category="$2"

    local template
    template=$(load_template "$template_file" "$category")

    echo "$template" | jq '.recommended_settings'
}

# =============================================================================
# CLI INTERFACE
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        chapter)
            build_chapter_illustration_prompt "${2:-a mysterious scene}" "${3:-}"
            ;;
        character)
            build_character_portrait_prompt "${2:-mysterious character}" "${3:-}" "${4:-}"
            ;;
        scene)
            build_scene_visualization_prompt "${2:-key moment in story}" "${3:-}"
            ;;
        toaster)
            build_toaster_prompt "${2:-contemplating existence}" "${3:-}"
            ;;
        avatar)
            build_persona_avatar_prompt "${2:-experimenter}"
            ;;
        icon)
            build_icon_prompt "${2:-dashboard icon}" "${3:-}"
            ;;
        settings)
            get_recommended_settings "$TEMPLATES_DIR/${2}.json" "${3}"
            ;;
        *)
            echo "Prompt Builder - Smart prompt generation from templates"
            echo ""
            echo "Usage:"
            echo "  $0 chapter <scene_description> [style]"
            echo "  $0 character <description> [expression] [art_style]"
            echo "  $0 scene <focal_point> [location]"
            echo "  $0 toaster <context> [personality]"
            echo "  $0 avatar <persona_name>"
            echo "  $0 icon <subject> [style]"
            echo "  $0 settings <template_file> <category>"
            echo ""
            echo "Examples:"
            echo "  $0 chapter 'a toaster standing in a rain-soaked alley'"
            echo "  $0 character 'chrome toaster with wise eyes' 'thoughtful gaze'"
            echo "  $0 toaster 'experiencing first sunrise' 'philosophical'"
            echo "  $0 avatar experimenter"
            ;;
    esac
fi
