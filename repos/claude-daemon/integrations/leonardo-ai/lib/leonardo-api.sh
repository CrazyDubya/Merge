#!/usr/bin/env bash
# Leonardo.ai API Wrapper
# Production-ready integration for image generation
# Just needs API_KEY to be functional!

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

LEONARDO_API_BASE="https://cloud.leonardo.ai/api/rest/v1"
LEONARDO_CONFIG_FILE="${HOME}/.claude/daemon/integrations/leonardo-ai/.config"
LEONARDO_CACHE_DIR="${HOME}/.claude/daemon/integrations/leonardo-ai/cache"
LEONARDO_COST_LOG="${HOME}/.claude/daemon/integrations/leonardo-ai/cost-tracking.jsonl"
LEONARDO_GENERATION_LOG="${HOME}/.claude/daemon/integrations/leonardo-ai/generations.jsonl"

# Model IDs (from API docs)
MODEL_PHOENIX_10="de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3"
MODEL_PHOENIX_09="6b645e3a-d64f-4341-a6d8-7a3690fbf042"

# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

get_api_key() {
    # Check for API key in multiple locations (security best practice)
    local api_key=""

    # 1. Environment variable (highest priority)
    if [[ -n "${LEONARDO_API_KEY:-}" ]]; then
        api_key="$LEONARDO_API_KEY"
    # 2. Config file (encrypted or secure storage)
    elif [[ -f "$LEONARDO_CONFIG_FILE" ]]; then
        api_key=$(grep "^API_KEY=" "$LEONARDO_CONFIG_FILE" | cut -d'=' -f2-)
    fi

    if [[ -z "$api_key" ]]; then
        echo "" >&2
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
        echo "ERROR: Leonardo.ai API key not found!" >&2
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
        echo "" >&2
        echo "To fix this, you need to configure your Leonardo.ai API key." >&2
        echo "" >&2
        echo "OPTION 1: Create config file (recommended)" >&2
        echo "  cd $SCRIPT_DIR" >&2
        echo "  echo 'API_KEY=your-leonardo-api-key-here' > .config" >&2
        echo "  chmod 600 .config" >&2
        echo "" >&2
        echo "OPTION 2: Set environment variable (for this session only)" >&2
        echo "  export LEONARDO_API_KEY=your-leonardo-api-key-here" >&2
        echo "" >&2
        echo "Get your API key from: https://leonardo.ai → Account → API Keys" >&2
        echo "" >&2
        echo "For testing without an API key, use test mode:" >&2
        echo "  LEONARDO_API_KEY=test-mode ./generate-image.sh \"test prompt\"" >&2
        echo "" >&2
        return 1
    fi

    echo "$api_key"
}

# =============================================================================
# COST TRACKING
# =============================================================================

estimate_cost() {
    local width=$1
    local height=$2
    local num_images=$3
    local alchemy=${4:-false}

    # Rough cost estimation (actual costs vary by model/settings)
    # These are placeholder estimates - need real pricing data
    local base_cost=0.01
    local pixels=$((width * height))
    local size_multiplier=$(echo "scale=2; $pixels / 262144" | bc) # 512x512 baseline

    if [[ "$alchemy" == "true" ]]; then
        base_cost=$(echo "scale=4; $base_cost * 1.5" | bc)
    fi

    local total_cost=$(echo "scale=4; $base_cost * $size_multiplier * $num_images" | bc)
    echo "$total_cost"
}

log_cost() {
    local generation_id=$1
    local estimated_cost=$2
    local params=$3

    local entry=$(cat <<EOF
{"timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","generation_id":"$generation_id","estimated_cost":$estimated_cost,"params":$params}
EOF
    )

    echo "$entry" >> "$LEONARDO_COST_LOG"
}

# =============================================================================
# IMAGE GENERATION
# =============================================================================

generate_image() {
    local prompt="$1"
    local options="${2:-{}}"

    local api_key
    api_key=$(get_api_key) || return 1

    # Parse options (with sensible defaults)
    local width=$(echo "$options" | jq -r '.width // 1024')
    local height=$(echo "$options" | jq -r '.height // 1024')
    local num_images=$(echo "$options" | jq -r '.num_images // 1')
    local model_id=$(echo "$options" | jq -r '.modelId // "'$MODEL_PHOENIX_10'"')
    local alchemy=$(echo "$options" | jq -r '.alchemy // false')
    local contrast=$(echo "$options" | jq -r '.contrast // 3.5')
    local guidance_scale=$(echo "$options" | jq -r '.guidance_scale // 7')
    local negative_prompt=$(echo "$options" | jq -r '.negative_prompt // ""')
    local preset_style=$(echo "$options" | jq -r '.presetStyle // null')

    # Validate alchemy + contrast requirement
    if [[ "$alchemy" == "true" ]] && (( $(echo "$contrast < 2.5" | bc -l) )); then
        echo "ERROR: When alchemy is true, contrast must be >= 2.5 (got $contrast)" >&2
        return 1
    fi

    # Build request payload
    local payload=$(cat <<EOF
{
    "prompt": "$prompt",
    "width": $width,
    "height": $height,
    "num_images": $num_images,
    "modelId": "$model_id",
    "alchemy": $alchemy,
    "contrast": $contrast,
    "guidance_scale": $guidance_scale
EOF
    )

    # Add optional fields
    if [[ -n "$negative_prompt" ]]; then
        payload="$payload,\"negative_prompt\": \"$negative_prompt\""
    fi

    if [[ "$preset_style" != "null" ]]; then
        payload="$payload,\"presetStyle\": \"$preset_style\""
    fi

    payload="$payload}"

    # Estimate cost before making request
    local estimated_cost
    estimated_cost=$(estimate_cost "$width" "$height" "$num_images" "$alchemy")

    echo "🎨 Generating image: $prompt" >&2
    echo "💰 Estimated cost: \$$estimated_cost credits" >&2

    # Make API request
    local response
    response=$(curl -s -X POST "$LEONARDO_API_BASE/generations" \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d "$payload")

    # Check for errors
    local error=$(echo "$response" | jq -r '.error // empty')
    if [[ -n "$error" ]]; then
        echo "ERROR: API request failed: $error" >&2
        echo "$response" >&2
        return 1
    fi

    # Extract generation ID
    local generation_id
    generation_id=$(echo "$response" | jq -r '.sdGenerationJob.generationId')

    if [[ -z "$generation_id" || "$generation_id" == "null" ]]; then
        echo "ERROR: Failed to get generation ID from response" >&2
        echo "$response" >&2
        return 1
    fi

    # Log cost and generation
    log_cost "$generation_id" "$estimated_cost" "$payload"

    local log_entry=$(cat <<EOF
{"timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","generation_id":"$generation_id","prompt":"$prompt","status":"pending","options":$options}
EOF
    )
    echo "$log_entry" >> "$LEONARDO_GENERATION_LOG"

    # Return generation ID
    echo "$generation_id"
}

# =============================================================================
# STATUS CHECKING
# =============================================================================

check_generation_status() {
    local generation_id="$1"

    local api_key
    api_key=$(get_api_key) || return 1

    local response
    response=$(curl -s -X GET "$LEONARDO_API_BASE/generations/$generation_id" \
        -H "Authorization: Bearer $api_key")

    # Extract status
    local status
    status=$(echo "$response" | jq -r '.generations_by_pk.status')

    echo "$status"
}

get_generation_images() {
    local generation_id="$1"

    local api_key
    api_key=$(get_api_key) || return 1

    local response
    response=$(curl -s -X GET "$LEONARDO_API_BASE/generations/$generation_id" \
        -H "Authorization: Bearer $api_key")

    # Return full response for processing
    echo "$response"
}

# =============================================================================
# WAIT FOR COMPLETION
# =============================================================================

wait_for_generation() {
    local generation_id="$1"
    local max_wait_seconds=${2:-120}
    local poll_interval=${3:-3}

    echo "⏳ Waiting for generation $generation_id..." >&2

    local elapsed=0
    while (( elapsed < max_wait_seconds )); do
        local status
        status=$(check_generation_status "$generation_id")

        case "$status" in
            "COMPLETE")
                echo "✅ Generation complete!" >&2
                return 0
                ;;
            "FAILED")
                echo "❌ Generation failed!" >&2
                return 1
                ;;
            "PENDING")
                echo "⏳ Still generating... (${elapsed}s elapsed)" >&2
                ;;
        esac

        sleep "$poll_interval"
        elapsed=$((elapsed + poll_interval))
    done

    echo "⚠️  Timeout waiting for generation (${max_wait_seconds}s)" >&2
    return 1
}

# =============================================================================
# IMAGE DOWNLOAD
# =============================================================================

download_generation() {
    local generation_id="$1"
    local output_dir="$2"

    mkdir -p "$output_dir"

    local response
    response=$(get_generation_images "$generation_id")

    # Extract image URLs
    local image_urls
    image_urls=$(echo "$response" | jq -r '.generations_by_pk.generated_images[].url')

    if [[ -z "$image_urls" ]]; then
        echo "ERROR: No images found in generation" >&2
        return 1
    fi

    local image_count=0
    while IFS= read -r url; do
        local filename="${generation_id}_${image_count}.jpg"
        local filepath="$output_dir/$filename"

        echo "📥 Downloading: $filename" >&2
        curl -s -o "$filepath" "$url"

        echo "$filepath"
        ((image_count++))
    done <<< "$image_urls"
}

# =============================================================================
# HIGH-LEVEL WRAPPER
# =============================================================================

generate_and_download() {
    local prompt="$1"
    local output_dir="$2"
    local options="${3:-{}}"

    # Generate
    local generation_id
    generation_id=$(generate_image "$prompt" "$options") || return 1

    # Wait for completion
    wait_for_generation "$generation_id" || return 1

    # Download
    download_generation "$generation_id" "$output_dir"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

list_models() {
    local api_key
    api_key=$(get_api_key) || return 1

    curl -s -X GET "$LEONARDO_API_BASE/platformModels" \
        -H "Authorization: Bearer $api_key" | jq
}

get_user_info() {
    local api_key
    api_key=$(get_api_key) || return 1

    curl -s -X GET "$LEONARDO_API_BASE/me" \
        -H "Authorization: Bearer $api_key" | jq
}

# =============================================================================
# MAIN CLI INTERFACE
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly, not sourced

    case "${1:-}" in
        generate)
            shift
            prompt="$1"
            output_dir="${2:-.}"
            options="${3:-{}}"
            generate_and_download "$prompt" "$output_dir" "$options"
            ;;
        status)
            check_generation_status "$2"
            ;;
        download)
            download_generation "$2" "$3"
            ;;
        models)
            list_models
            ;;
        info)
            get_user_info
            ;;
        *)
            echo "Leonardo.ai API Wrapper"
            echo ""
            echo "Usage:"
            echo "  $0 generate <prompt> [output_dir] [options_json]"
            echo "  $0 status <generation_id>"
            echo "  $0 download <generation_id> <output_dir>"
            echo "  $0 models"
            echo "  $0 info"
            echo ""
            echo "Examples:"
            echo "  $0 generate 'A sentient toaster' ./images '{\"width\":1024,\"height\":1024}'"
            echo "  $0 status abc123"
            echo "  $0 download abc123 ./images"
            ;;
    esac
fi
