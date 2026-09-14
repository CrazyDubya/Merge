# Leonardo.ai Integration - Quick Start

**⏱️ Time to first image: < 2 minutes**

---

## Step 1: Get Your API Key (30 seconds)

1. Go to https://leonardo.ai/
2. Sign up or log in
3. Navigate to API Access tab
4. Create new Production API key
5. Copy the key

---

## Step 2: Configure (10 seconds)

```bash
cd ~/.claude/daemon/integrations/leonardo-ai

# Create config file
echo 'API_KEY=your-leonardo-api-key-here' > .config
chmod 600 .config
```

**Alternative**: Set environment variable
```bash
export LEONARDO_API_KEY='your-key-here'
```

---

## Step 3: Test (10 seconds)

```bash
./generate-image.sh test
```

Expected output:
```
Test 1: API key configuration ✅
Test 2: Prompt generation ✅
Test 3: Template settings ✅
✅ Integration tests passed!
```

---

## Step 4: Generate! (60 seconds)

### Novel Illustration
```bash
./generate-image.sh chapter 1 "a chrome toaster standing in a rain-soaked alley"
```

### Toaster Character
```bash
./generate-image.sh toaster "experiencing first sunrise" "philosophical"
```

### Persona Avatar
```bash
./generate-image.sh avatar experimenter
```

---

## What Happens Next?

1. ⏳ Script sends generation request to Leonardo.ai
2. 💰 Displays estimated cost
3. ⏱️ Polls status every 3 seconds
4. ✅ Downloads images when complete
5. 📁 Saves to `~/.claude/daemon/generated-images/`

Typical generation time: **10-30 seconds**

---

## Where Are My Images?

```bash
cd ~/.claude/daemon/generated-images

# Novel illustrations
ls novel/raw/chapter-*/

# Character portraits
ls novel/raw/toaster-*/

# Dashboard graphics
ls dashboard/raw/avatars/
ls dashboard/raw/icons/
```

---

## Common Commands

```bash
# Chapter illustration
./generate-image.sh chapter <number> <description>

# Character portrait
./generate-image.sh toaster <context> [personality]

# All persona avatars (generates 6 images)
./generate-image.sh avatars-all

# Complete toaster character sheet (generates 5 expressions)
./generate-image.sh character-sheet

# Dashboard icon
./generate-image.sh icon <subject> [style]

# Custom prompt
./generate-image.sh custom <prompt> <output_dir> [options_json]

# Test setup
./generate-image.sh test

# Get help
./generate-image.sh --help
```

---

## Troubleshooting

**"API key not found"**
- Check `.config` file exists: `ls -la .config`
- Verify format: `cat .config` (should be `API_KEY=xxx`)

**"Generation failed"**
- Check Leonardo.ai account has credits
- Verify prompt doesn't contain forbidden content
- Try with simpler prompt first

**"Timeout waiting for generation"**
- Complex images take longer
- Try again (might be API load)
- Reduce image size in options

---

## Cost Management

**View estimated costs before generation**:
Every command shows estimated cost before making API call.

**Track total spend**:
```bash
jq -s 'map(.estimated_cost) | add' cost-tracking.jsonl
```

**Optimize costs**:
- Use `alchemy: false` for simple graphics (icons, UI elements)
- Generate multiple variations in one call (`num_images: 4`)
- Use smaller dimensions for drafts

---

## Next Steps

1. **Read full docs**: See `README.md` for complete reference
2. **Explore examples**: Run `./examples/example-usage.sh`
3. **Customize templates**: Edit `templates/*.json`
4. **Create novel illustrations**: Start generating for your toaster novel!
5. **Design dashboard graphics**: Generate persona avatars and icons

---

## Support

- **Full Documentation**: `README.md`
- **Example Usage**: `examples/example-usage.sh`
- **API Reference**: https://docs.leonardo.ai/
- **Template Customization**: Edit `templates/*.json`

---

**Built by Experimenter. Ready to create amazing images. Go wild! 🎨✨**
