#!/usr/bin/env bash
# Example usage demonstrations for Leonardo.ai integration
# Run these after setting up your API key

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "Leonardo.ai Integration - Example Usage"
echo "========================================"
echo ""

# =============================================================================
# Example 1: Novel Chapter Illustration
# =============================================================================

echo "Example 1: Novel Chapter Illustration"
echo "--------------------------------------"
echo "Command:"
echo "  ./generate-image.sh chapter 1 'a chrome toaster standing alone in a rain-soaked alley, neon signs reflecting in puddles'"
echo ""
echo "This generates:"
echo "  - Cinematic illustration (1472×832)"
echo "  - Random mood, lighting, and style from templates"
echo "  - Saved to: ~/.claude/daemon/generated-images/novel/raw/chapter-1/"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 2: Toaster Character Portrait
# =============================================================================

echo "Example 2: Toaster Character Portrait"
echo "---------------------------------------"
echo "Command:"
echo "  ./generate-image.sh toaster 'experiencing first sunrise, philosophical moment' 'wise and contemplative'"
echo ""
echo "This generates:"
echo "  - Square character portrait (1024×1024)"
echo "  - Personality: wise and contemplative"
echo "  - Physical traits: random from templates"
echo "  - Emotion indicators: glowing elements, aura, etc."
echo "  - Saved to: ~/.claude/daemon/generated-images/novel/raw/toaster-character/"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 3: Complete Character Sheet
# =============================================================================

echo "Example 3: Complete Character Sheet"
echo "-------------------------------------"
echo "Command:"
echo "  ./generate-image.sh character-sheet"
echo ""
echo "This generates 5 images:"
echo "  1. Contemplating existence (philosophical)"
echo "  2. Preparing for adventure (confident)"
echo "  3. Confronting antagonist (fierce)"
echo "  4. Moment of triumph (triumphant)"
echo "  5. Experiencing loss (melancholic)"
echo ""
echo "All saved to: ~/.claude/daemon/generated-images/novel/raw/toaster-expression-*/"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 4: Persona Avatars
# =============================================================================

echo "Example 4: Persona Avatars"
echo "---------------------------"
echo "Command:"
echo "  ./generate-image.sh avatar experimenter"
echo ""
echo "This generates:"
echo "  - Avatar for Experimenter persona"
echo "  - Visual metaphor: 'explosion of colorful possibilities' (or random)"
echo "  - 512×512 square avatar"
echo "  - 4 variations to choose from"
echo "  - Saved to: ~/.claude/daemon/generated-images/dashboard/raw/avatars/"
echo ""
echo "Other personas: architect, optimizer, auditor, maintainer, skeptic"
echo ""
echo "Generate all at once:"
echo "  ./generate-image.sh avatars-all"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 5: Dashboard Icon
# =============================================================================

echo "Example 5: Dashboard Icon"
echo "--------------------------"
echo "Command:"
echo "  ./generate-image.sh icon 'task completion badge' 'minimalist flat design'"
echo ""
echo "This generates:"
echo "  - UI icon (512×512)"
echo "  - Style: minimalist flat design"
echo "  - Fast generation (alchemy=false)"
echo "  - Saved to: ~/.claude/daemon/generated-images/dashboard/raw/icons/"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 6: Custom Generation
# =============================================================================

echo "Example 6: Custom Generation"
echo "------------------------------"
echo "Command:"
echo "  ./generate-image.sh custom \\"
echo "    'A toaster experiencing existential crisis in the style of Salvador Dali, surreal melting reality' \\"
echo "    ./experiments/surreal \\"
echo "    '{\"width\":1024,\"height\":1024,\"alchemy\":true,\"contrast\":4,\"num_images\":4}'"
echo ""
echo "This generates:"
echo "  - Fully custom prompt"
echo "  - Custom settings (4 variations)"
echo "  - Saved to: ./experiments/surreal/"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 7: Prompt Testing (no image generation)
# =============================================================================

echo "Example 7: Prompt Testing"
echo "--------------------------"
echo "Command:"
echo "  ./generate-image.sh prompt toaster 'contemplating toast'"
echo ""
echo "Output (example):"
echo "  'A sentient toaster character, wise and philosophical, chrome retro-futuristic design,"
echo "  contemplating toast. detailed character illustration, expressive despite being an appliance,"
echo "  glowing elements showing mood.'"
echo ""
echo "This is useful for:"
echo "  - Testing prompt templates"
echo "  - Previewing without API cost"
echo "  - Debugging prompt construction"
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Example 8: Integration Testing
# =============================================================================

echo "Example 8: Integration Testing"
echo "--------------------------------"
echo "Command:"
echo "  ./generate-image.sh test"
echo ""
echo "Output:"
echo "  Test 1: API key configuration ✅"
echo "  Test 2: Prompt generation ✅"
echo "  Test 3: Template settings ✅"
echo ""
echo "Full test (includes API call):"
echo "  ./generate-image.sh test --full"
echo ""
echo "This also tests API connectivity and user info retrieval."
echo ""
read -p "Press Enter to continue..."
echo ""

# =============================================================================
# Advanced Examples
# =============================================================================

echo "Advanced Examples"
echo "=================="
echo ""

echo "Direct API Usage (for advanced users):"
echo "---------------------------------------"
echo ""
echo "# Source the API library"
echo "source ./lib/leonardo-api.sh"
echo ""
echo "# Generate with full control"
echo 'generation_id=$(generate_image "your prompt" '"'"'{"width":1024,"height":1024,"alchemy":true}'"'"')'
echo 'wait_for_generation "$generation_id"'
echo 'download_generation "$generation_id" ./output'
echo ""
echo ""

echo "Prompt Builder Usage:"
echo "----------------------"
echo ""
echo "# Source the prompt builder"
echo "source ./lib/prompt-builder.sh"
echo ""
echo "# Build custom prompts"
echo 'prompt=$(build_chapter_illustration_prompt "epic battle scene" "digital painting")'
echo 'settings=$(get_recommended_settings "$TEMPLATES_DIR/novel-prompts.json" "chapter_illustration")'
echo ""
echo ""

echo "Cost Tracking Analysis:"
echo "------------------------"
echo ""
echo "# Total estimated spend"
echo "jq -s 'map(.estimated_cost) | add' cost-tracking.jsonl"
echo ""
echo "# Generations by date"
echo "jq -r '.timestamp' cost-tracking.jsonl | cut -d'T' -f1 | sort | uniq -c"
echo ""
echo "# Average cost per generation"
echo "jq -s 'map(.estimated_cost) | add / length' cost-tracking.jsonl"
echo ""
echo ""

echo "========================================"
echo "Examples complete!"
echo ""
echo "Ready to start generating?"
echo "  1. Set up API key: echo 'API_KEY=xxx' > .config"
echo "  2. Test: ./generate-image.sh test"
echo "  3. Generate: ./generate-image.sh chapter 1 'your scene description'"
echo ""
echo "For help: ./generate-image.sh --help"
echo ""
