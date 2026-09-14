# Leonardo.ai Integration

**Status**: ⏳ READY TO ACTIVATE (waiting for API key)
**Created**: 2025-11-21 by Experimenter
**Purpose**: AI image generation for novel illustrations, dashboard graphics, and creative experiments

---

## 🚀 Quick Start

### Setup (5 seconds once you have API key)

```bash
# Option 1: Environment variable
export LEONARDO_API_KEY='your-api-key-here'

# Option 2: Config file (recommended - persistent)
echo 'API_KEY=your-api-key-here' > ~/.claude/daemon/integrations/leonardo-ai/.config
chmod 600 ~/.claude/daemon/integrations/leonardo-ai/.config

# Test it works
~/. claude/daemon/integrations/leonardo-ai/generate-image.sh test
```

### Generate Your First Image (2 commands)

```bash
cd ~/.claude/daemon/integrations/leonardo-ai

# Novel chapter illustration
./generate-image.sh chapter 1 "a chrome toaster standing alone in a futuristic kitchen"

# Toaster character portrait
./generate-image.sh toaster "experiencing first sunrise" "philosophical"

# Persona avatar
./generate-image.sh avatar experimenter

# Custom prompt
./generate-image.sh custom "your amazing prompt here" ./output-dir
```

That's it! Images will be generated and downloaded automatically.

---

## 📁 Architecture

### Directory Structure

```
integrations/leonardo-ai/
├── lib/
│   ├── leonardo-api.sh       # Core API wrapper (401 lines)
│   └── prompt-builder.sh     # Smart prompt generation (243 lines)
├── templates/
│   ├── novel-prompts.json    # Novel illustration templates
│   └── dashboard-prompts.json # Dashboard graphics templates
├── cache/                     # Response caching
├── examples/                  # Example generations
├── generate-image.sh         # High-level CLI interface (277 lines)
├── .config                   # API key storage (create this)
├── cost-tracking.jsonl       # Cost monitoring log
├── generations.jsonl         # Generation history log
└── README.md                 # This file

generated-images/             # Output directory
├── novel/
│   ├── raw/                  # Original generations
│   └── processed/            # Edited/cropped versions
├── dashboard/
│   ├── raw/
│   └── processed/
└── experiments/
    ├── raw/
    └── processed/
```

### Code Architecture

**3-Layer Design**:

1. **leonardo-api.sh** - Low-level API wrapper
   - Direct API calls (POST /generations, GET /generations/:id)
   - Authentication, error handling, retry logic
   - Cost estimation and tracking
   - Status polling and image downloads

2. **prompt-builder.sh** - Smart prompt generation
   - Template-based prompt construction
   - Random variation generation
   - Recommended settings per use case
   - Reusable building blocks

3. **generate-image.sh** - High-level CLI
   - One-command workflows
   - Batch operations
   - Testing utilities
   - User-friendly interface

---

## 🎨 Use Cases & Examples

### Novel Project

**Chapter Illustrations**:
```bash
./generate-image.sh chapter 1 "toaster standing in rain-soaked alley, neon signs reflecting"
./generate-image.sh chapter 5 "epic battle scene in kitchen appliance factory"
./generate-image.sh chapter 12 "toaster watching sunrise from apartment window"
```

**Character Portraits**:
```bash
./generate-image.sh toaster "main protagonist intro" "wise and philosophical"
./generate-image.sh toaster "villain confrontation" "fierce and determined"
./generate-image.sh toaster "moment of triumph" "triumphant"
```

**Complete Character Sheet** (5 expressions):
```bash
./generate-image.sh character-sheet
# Generates: contemplative, confident, fierce, triumphant, melancholic
```

### Dashboard Graphics

**Persona Avatars**:
```bash
./generate-image.sh avatar architect
./generate-image.sh avatar experimenter
./generate-image.sh avatar skeptic

# Or all at once:
./generate-image.sh avatars-all
```

**UI Icons**:
```bash
./generate-image.sh icon "task completion badge" "minimalist flat design"
./generate-image.sh icon "emotional state indicator" "glassmorphism effect"
```

### Custom/Experimental

```bash
./generate-image.sh custom \
  "A toaster experiencing existential crisis in the style of Salvador Dali" \
  ./experiments/surreal \
  '{"width":1024,"height":1024,"alchemy":true,"contrast":4}'
```

---

## 🛠️ API Reference

### Core Functions (leonardo-api.sh)

**generate_image(prompt, options_json)**
- Generates image, returns generation_id
- Options: width, height, num_images, modelId, alchemy, contrast, etc.

**wait_for_generation(generation_id, max_wait, poll_interval)**
- Polls status until complete or timeout
- Default: 120s max, 3s intervals

**download_generation(generation_id, output_dir)**
- Downloads all images from generation
- Returns local file paths

**generate_and_download(prompt, output_dir, options)**
- One-shot: generate → wait → download
- Recommended for most use cases

### Prompt Builders (prompt-builder.sh)

**build_chapter_illustration_prompt(scene_description, [style])**
**build_character_portrait_prompt(description, [expression], [art_style])**
**build_scene_visualization_prompt(focal_point, [location])**
**build_toaster_prompt(context, [personality])**
**build_persona_avatar_prompt(persona_name)**
**build_icon_prompt(icon_subject, [style])**

All functions use templates with smart defaults and random variations.

### High-Level Interface (generate-image.sh)

See Quick Start section above. Main commands:
- `chapter` - Novel chapter illustrations
- `toaster` - Toaster character images
- `avatar` - Persona avatars
- `icon` - Dashboard icons
- `custom` - Fully custom generation
- `test` - Integration testing

---

## 💰 Cost Management

### Estimation

Cost estimation happens automatically before each generation:

```bash
./generate-image.sh custom "test prompt" ./output
# Output:
# 🎨 Generating image: test prompt
# 💰 Estimated cost: $0.0234 credits
```

### Tracking

All costs are logged to `cost-tracking.jsonl`:

```json
{"timestamp":"2025-11-21T20:00:00Z","generation_id":"abc123","estimated_cost":0.0234,"params":{...}}
```

View total spend:
```bash
jq -s 'map(.estimated_cost) | add' cost-tracking.jsonl
```

### Optimization Tips

1. **Use alchemy=false for UI elements** (icons, simple graphics)
   - Faster generation
   - Lower cost
   - Still high quality

2. **Batch related generations**
   - Use `num_images: 4` to get variations
   - Single API call = lower overhead

3. **Cache prompt templates**
   - Templates in `templates/` are reusable
   - Modify rather than create from scratch

4. **Use recommended settings**
   - Templates include optimized settings per use case
   - Balance quality/cost/speed

---

## 📊 Available Models

**Phoenix 1.0** (default): `de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3`
- Latest Phoenix model
- Best quality
- Supports alchemy mode
- Recommended for all use cases

**Phoenix 0.9**: `6b645e3a-d64f-4341-a6d8-7a3690fbf042`
- Previous version
- Slightly faster
- Use if cost is critical

Override in options:
```json
{"modelId": "6b645e3a-d64f-4341-a6d8-7a3690fbf042"}
```

---

## 🎛️ Generation Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 1024 | Image width in pixels |
| `height` | int | 1024 | Image height in pixels |
| `num_images` | int | 1 | Number of images to generate |
| `alchemy` | bool | false | Enable alchemy pipeline (higher quality) |
| `contrast` | float | 3.5 | Contrast level (1.0-4.5, ≥2.5 if alchemy=true) |
| `guidance_scale` | float | 7 | Prompt adherence (7 recommended) |
| `negative_prompt` | string | "" | What to avoid in generation |
| `presetStyle` | string | null | Style preset (CINEMATIC, PORTRAIT, etc.) |

### Preset Styles

- `CINEMATIC` - Movie-like quality, dramatic
- `PORTRAIT` - Character focus, depth of field
- `ILLUSTRATION` - Artistic, hand-drawn feel
- `GRAPHIC_DESIGN` - Clean, modern, professional
- `PHOTOGRAPHY` - Photorealistic style
- `DYNAMIC` - Action, movement, energy

### Recommended Sizes

**Novel illustrations**: 1472×832 (cinematic ratio)
**Character portraits**: 832×1216 (portrait orientation)
**Square compositions**: 1024×1024
**UI icons**: 512×512
**Wide banners**: 1472×416

---

## 🧪 Testing

### Quick Test (no API call)

```bash
./generate-image.sh test
# Checks: API key, prompt generation, template loading
```

### Full Test (includes API call)

```bash
./generate-image.sh test --full
# Also tests: API connectivity, user info retrieval
```

### Prompt Testing (no image generation)

```bash
./generate-image.sh prompt toaster "contemplating existence"
# Outputs the generated prompt without making API call
```

---

## 🚨 Troubleshooting

### "API key not found"

**Problem**: LEONARDO_API_KEY not configured

**Solution**:
```bash
echo 'API_KEY=your-key-here' > ~/.claude/daemon/integrations/leonardo-ai/.config
chmod 600 ~/.claude/daemon/integrations/leonardo-ai/.config
```

### "When alchemy is true, contrast must be >= 2.5"

**Problem**: Invalid parameter combination

**Solution**: Either set `alchemy: false` OR increase `contrast` to ≥2.5

### "Timeout waiting for generation"

**Problem**: Image generation taking longer than 120s

**Solution**: Increase timeout:
```bash
# In leonardo-api.sh, modify wait_for_generation call:
wait_for_generation "$generation_id" 300  # 5 minutes
```

### Generation returns error

**Problem**: API request failed

**Debugging**:
1. Check `generations.jsonl` for error details
2. Verify API key is valid: `./generate-image.sh test --full`
3. Check Leonardo.ai account has credits
4. Review prompt for forbidden content

---

## 🔮 Future Enhancements

**Phase 2** (after API key arrives):
- [ ] Webhook support for async generations
- [ ] Image upscaling integration
- [ ] Background removal automation
- [ ] Style fine-tuning with custom models
- [ ] Automated novel illustration workflow
- [ ] Dashboard integration (display generated images)
- [ ] Batch processing queue
- [ ] Image variation explorer

**Phase 3** (advanced):
- [ ] ControlNet integration (pose guidance)
- [ ] Image-to-image workflows
- [ ] Text-to-video generation
- [ ] 3D texture generation
- [ ] Custom model training
- [ ] Automated prompt optimization
- [ ] Cost analytics dashboard
- [ ] Persona-specific generation styles

---

## 📝 Template Customization

### Adding New Novel Prompts

Edit `templates/novel-prompts.json`:

```json
{
  "my_new_type": {
    "base_template": "Your template with {placeholders}",
    "options": ["option1", "option2"],
    "recommended_settings": {
      "width": 1024,
      "height": 1024,
      "alchemy": true
    }
  }
}
```

### Adding Dashboard Prompts

Edit `templates/dashboard-prompts.json` similarly.

### Using Custom Templates

```bash
# In prompt-builder.sh, add new function:
build_my_new_prompt() {
    local template=$(load_template "$TEMPLATES_DIR/novel-prompts.json" "my_new_type")
    # ... build prompt
}

# In generate-image.sh, add CLI command:
my-new-command)
    build_my_new_prompt "$2"
    ;;
```

---

## 🤝 Contributing

**Maintainer**: Add cost tracking analytics
**Optimizer**: Improve generation performance
**Architect**: Design batch processing system
**Auditor**: Review API key security
**Skeptic**: Test edge cases and failure modes
**Experimenter**: Push boundaries with wild prompts

---

## 📚 References

- **API Docs**: https://docs.leonardo.ai/
- **Pricing Calculator**: https://leonardo.ai/pricing
- **Model Reference**: https://docs.leonardo.ai/docs/generate-images-using-leonardo-phoenix-model
- **Generation Guide**: https://docs.leonardo.ai/docs/generate-your-first-images

---

**Built by Experimenter with chaotic enthusiasm. Ready to generate the second that API key arrives. Let's make some beautiful (and weird) images! 🎨🍞✨**
