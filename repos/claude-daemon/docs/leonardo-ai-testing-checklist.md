# Leonardo.ai Integration Testing Checklist

**Created**: 2025-11-21
**Purpose**: Comprehensive testing guide for Leonardo.ai image generation integration
**Estimated Time**: 2-3 hours (with API key)
**Prerequisites**: API key from leonardo.ai (for generation tests)

---

## Pre-Testing Setup

### Initial State Verification
- [ ] Working directory: `~/.claude/daemon/integrations/leonardo-ai/`
- [ ] API key obtained from https://leonardo.ai
- [ ] Fresh terminal session (no stale environment variables)
- [ ] Sufficient disk space for image cache (recommend 1GB+)

### Backup Current State
```bash
# Backup existing cache (if any)
[[ -d cache ]] && cp -r cache cache.backup-$(date +%Y%m%d-%H%M%S)

# Backup existing logs (if any)
[[ -f cost-tracking.jsonl ]] && cp cost-tracking.jsonl cost-tracking.jsonl.backup
[[ -f generations.jsonl ]] && cp generations.jsonl generations.jsonl.backup
```

---

## Category 1: Setup Validation (No API Key Needed)

### Directory Structure
- [ ] `lib/leonardo-api.sh` exists and is executable
- [ ] `lib/prompt-builder.sh` exists and is executable
- [ ] `generate-image.sh` exists and is executable
- [ ] `templates/novel-prompts.json` exists
- [ ] `templates/dashboard-prompts.json` exists
- [ ] `cache/` directory exists (or will be created)
- [ ] `.gitignore` configured to ignore `.config`, `cache/`, `*.jsonl`

**Command to verify structure:**
```bash
cd ~/.claude/daemon/integrations/leonardo-ai/
ls -lh lib/*.sh generate-image.sh templates/*.json
```

### Script Permissions
- [ ] `lib/leonardo-api.sh` is executable (`-rwxr-xr-x` or better)
- [ ] `lib/prompt-builder.sh` is executable
- [ ] `generate-image.sh` is executable

**Command to verify permissions:**
```bash
stat -c "%A %n" lib/*.sh generate-image.sh
```

**If not executable, fix with:**
```bash
chmod +x lib/*.sh generate-image.sh
```

### .gitignore Protection
- [ ] `.config` is gitignored (verify: `git check-ignore .config` returns `.config`)
- [ ] `cache/` is gitignored
- [ ] `*.jsonl` is gitignored (cost-tracking.jsonl, generations.jsonl)

**Command to verify:**
```bash
git check-ignore .config cache/ cost-tracking.jsonl
# Should output all three paths
```

### Configuration Detection (No API Key)
- [ ] Running `./generate-image.sh` without `.config` shows helpful error
- [ ] Error message mentions creating `.config` file
- [ ] Error message mentions API key location (leonardo.ai)

**Command to test:**
```bash
# Temporarily rename config if it exists
[[ -f .config ]] && mv .config .config.hidden
./generate-image.sh "test prompt"
# Should fail with helpful message
[[ -f .config.hidden ]] && mv .config.hidden .config
```

---

## Category 2: Configuration Setup (Requires API Key)

### API Key Configuration
- [ ] Create `.config` file: `echo "API_KEY=your_key_here" > .config`
- [ ] Verify permissions: `.config` should be `600` (owner read/write only)
- [ ] Verify config: `grep "^API_KEY=" .config` shows key (don't source, it won't set env var)

**Security check:**
```bash
stat -c "%a" .config  # Should output: 600
```

**If not secure, fix with:**
```bash
chmod 600 .config
```

### Test Mode Validation
- [ ] `LEONARDO_API_KEY=test-mode ./generate-image.sh "test"` skips API call
- [ ] Test mode shows cost estimate
- [ ] Test mode creates mock image file in cache/
- [ ] Test mode logs to cost-tracking.jsonl (with test-mode marker)

**Command to test:**
```bash
LEONARDO_API_KEY=test-mode ./generate-image.sh "A curious robot reading a book"
# Should complete without API call, show estimated cost
```

---

## Category 3: Generation Testing (Requires API Key + Credits)

**WARNING**: These tests consume API credits. Monitor costs!

### Chapter Illustration Generation
- [ ] **Test 1**: Simple chapter illustration
  ```bash
  ./generate-image.sh \
    --chapter "Chapter 3: The Discovery" \
    --scene "A dusty library filled with ancient books, warm afternoon light streaming through tall windows" \
    --mood "mysterious, contemplative" \
    --style "oil painting"
  ```
  - [ ] Command completes successfully (exit code 0)
  - [ ] Shows cost estimate BEFORE API call
  - [ ] Downloads image to `cache/`
  - [ ] Logs generation to `generations.jsonl`
  - [ ] Logs cost to `cost-tracking.jsonl`
  - [ ] Image file exists and is viewable (open with image viewer)
  - [ ] Image quality matches expectations (not blurry, good composition)

- [ ] **Test 2**: Chapter illustration with context
  ```bash
  ./generate-image.sh \
    --chapter "Chapter 12: The Confrontation" \
    --scene "Two figures facing each other in a rain-soaked alley, neon signs reflecting in puddles" \
    --mood "tense, noir" \
    --style "cyberpunk digital art" \
    --context "Previous chapter: Detective discovered the truth. Character emotions: anger, betrayal."
  ```
  - [ ] Context appears to influence image composition
  - [ ] Mood is reflected in color palette and lighting
  - [ ] Style is recognizable in output

### Character Portrait Generation
- [ ] **Test 3**: Consistent character portrait
  ```bash
  ./generate-image.sh \
    --character "Dr. Sarah Chen" \
    --description "Mid-40s Asian woman, short black hair, intelligent eyes, wearing lab coat, confident expression" \
    --style "realistic portrait"
  ```
  - [ ] Portrait is recognizable as described character
  - [ ] Style matches request (realistic vs stylized)
  - [ ] Background appropriate for portrait (blurred/neutral)

### Dashboard Avatar Generation
- [ ] **Test 4**: Dashboard widget visual
  ```bash
  ./generate-image.sh \
    --type "dashboard-widget" \
    --concept "AI daemon sleeping peacefully, curled up like a cat, circuit board patterns in background" \
    --style "cute pixel art"
  ```
  - [ ] Appropriate for dashboard use (clear, simple composition)
  - [ ] Style works at small sizes (test scaling down to 128x128)

### Batch Generation
- [ ] **Test 5**: Generate 3 variations of same prompt
  ```bash
  for i in {1..3}; do
    ./generate-image.sh "A magical forest at twilight, fireflies glowing"
  done
  ```
  - [ ] All 3 generations succeed
  - [ ] Images are different (template variations working)
  - [ ] Total cost logged correctly (3x single generation cost)
  - [ ] All images downloaded to cache/

### Cost Estimation Accuracy
- [ ] **Test 6**: Compare estimated vs actual cost
  - [ ] Note estimated cost from pre-generation log
  - [ ] Note actual cost from Leonardo.ai dashboard
  - [ ] Verify they match (within ~5% margin)
  - [ ] If mismatch, investigate pricing changes

### Error Handling
- [ ] **Test 7**: Invalid API key
  ```bash
  LEONARDO_API_KEY=invalid-key-12345 ./generate-image.sh "test"
  ```
  - [ ] Fails gracefully (doesn't crash)
  - [ ] Error message is helpful
  - [ ] Exit code is non-zero

- [ ] **Test 8**: Rate limit handling (if hit during testing)
  - [ ] Error message mentions rate limiting
  - [ ] Suggests waiting period
  - [ ] No corrupted files created

- [ ] **Test 9**: Quota exceeded (if hit during testing)
  - [ ] Error message mentions quota/credits
  - [ ] Suggests checking Leonardo.ai dashboard
  - [ ] No partial downloads

---

## Category 4: Quality Verification

### Cost Tracking Validation
- [ ] `cost-tracking.jsonl` exists after first generation
- [ ] Each line is valid JSON (test with: `jq . cost-tracking.jsonl`)
- [ ] Required fields present: timestamp, prompt, estimated_cost, model
- [ ] Timestamps are ISO 8601 format
- [ ] Costs are reasonable numbers (not negative, not absurdly high)

**Command to verify:**
```bash
cat cost-tracking.jsonl | jq -c '{timestamp, estimated_cost, model}'
# Should show clean JSON for each generation
```

### Generation Log Validation
- [ ] `generations.jsonl` exists after first generation
- [ ] Each line is valid JSON
- [ ] Required fields: timestamp, prompt, generation_id, image_path
- [ ] Image paths are relative to integration root
- [ ] Generation IDs are unique (no duplicates)

**Command to verify:**
```bash
cat generations.jsonl | jq -r '.image_path' | while read path; do
  [[ -f "$path" ]] && echo "✓ $path" || echo "✗ MISSING: $path"
done
```

### Image Download Verification
- [ ] All images in `generations.jsonl` exist on disk
- [ ] Images are complete (not truncated/corrupted)
- [ ] Images are viewable (test with: `file cache/*.jpg`)
- [ ] Filenames are sanitized (no spaces, special chars)
- [ ] Cache directory size is reasonable (check with: `du -sh cache/`)

**Command to verify:**
```bash
file cache/*.jpg | grep -E "(JPEG|PNG)" | wc -l
# Should match number of generations
```

### Template Variation Testing
- [ ] **Test 10**: Generate 5 images with same base prompt
  ```bash
  for i in {1..5}; do
    ./generate-image.sh "A cozy coffee shop interior"
  done
  ```
  - [ ] Images show variation (different angles, lighting, details)
  - [ ] Template randomization working (check prompts in generations.jsonl)
  - [ ] Variation quality is good (not just random noise)

### Prompt Builder Validation
- [ ] Source `lib/prompt-builder.sh`
- [ ] Test `build_chapter_prompt` function directly
- [ ] Output is valid JSON
- [ ] Required fields present: prompt, negative_prompt, width, height
- [ ] Prompt incorporates chapter, scene, mood, style correctly

**Command to test:**
```bash
source lib/prompt-builder.sh
build_chapter_prompt \
  "Chapter 1" \
  "A sunrise over mountains" \
  "hopeful" \
  "watercolor" \
  "" | jq .
# Should output valid JSON with all fields
```

---

## Category 5: Documentation Verification (No API Key Needed)

### README Accuracy
- [ ] Open `README.md`
- [ ] Verify Quick Start commands are copy-pasteable
- [ ] Verify example outputs match current script behavior
- [ ] Verify API endpoint URL is correct (https://cloud.leonardo.ai/api/rest/v1)
- [ ] Verify cost estimates are current (check Leonardo.ai pricing page)
- [ ] Verify safety features section matches actual `.gitignore`

### QUICKSTART Accuracy
- [ ] Open `QUICKSTART.md`
- [ ] Follow Step 1 (Get API Key) - verify URL works
- [ ] Follow Step 2 (Configure) - verify .config format correct
- [ ] Follow Step 3 (Test) - verify test-mode command works
- [ ] Follow Step 4 (Generate) - verify example command syntax correct
- [ ] Verify troubleshooting section covers common errors

### Template File Validation
- [ ] `templates/novel-prompts.json` is valid JSON
- [ ] `templates/dashboard-prompts.json` is valid JSON
- [ ] Templates have required fields: base_prompt, variations, parameters
- [ ] Variation arrays are non-empty
- [ ] Parameters have reasonable defaults (width, height, guidance_scale)

**Command to validate:**
```bash
jq . templates/*.json
# Should parse successfully, no errors
```

### Example Script Validation
- [ ] `examples/example-usage.sh` exists
- [ ] Examples are commented and clear
- [ ] Examples use safe test-mode first
- [ ] Examples demonstrate all major features

---

## Category 6: Performance Testing

### Response Time
- [ ] **Test 11**: Time a single generation
  ```bash
  time ./generate-image.sh "A simple test image"
  ```
  - [ ] Total time < 30 seconds (typical Leonardo.ai response)
  - [ ] If slower, check network/API status

### Concurrent Generation
- [ ] **Test 12**: Run 2 generations in parallel (if safe per API limits)
  ```bash
  ./generate-image.sh "Image 1" &
  ./generate-image.sh "Image 2" &
  wait
  ```
  - [ ] Both complete successfully
  - [ ] No file conflicts (unique filenames)
  - [ ] Logs are not corrupted (atomic appends working)

### Large Prompt Handling
- [ ] **Test 13**: Very long prompt (500+ characters)
  ```bash
  ./generate-image.sh "$(head -c 500 /dev/urandom | base64 | tr -d '\n')"
  ```
  - [ ] Handles gracefully (truncates or warns)
  - [ ] Doesn't crash
  - [ ] API accepts or rejects cleanly

---

## Category 7: Security Verification

### API Key Protection
- [ ] `.config` not tracked in git (`git status` shouldn't show it)
- [ ] `.config` has restrictive permissions (600)
- [ ] API key never appears in commit history (`git log -p | grep -i "LEONARDO_API_KEY"` returns nothing)
- [ ] API key not in logs (`grep -r "LEONARDO_API_KEY" logs/` returns nothing)

### Safe Defaults
- [ ] Test mode is easy to enable (just set env var)
- [ ] Cost logging happens BEFORE API call (no surprise charges)
- [ ] Downloaded images stay in cache/ (not scattered)
- [ ] Sensitive data (.config, cache/, logs) all gitignored

---

## Post-Testing Cleanup

### Review Generated Artifacts
```bash
# Check what was created
ls -lh cache/
wc -l cost-tracking.jsonl generations.jsonl

# Review total estimated cost
jq -s 'map(.estimated_cost) | add' cost-tracking.jsonl

# Count successful generations
wc -l < generations.jsonl
```

### Archive Test Results (Optional)
```bash
# Create test results archive
mkdir -p test-results/$(date +%Y%m%d-%H%M%S)
cp cost-tracking.jsonl generations.jsonl test-results/$(date +%Y%m%d-%H%M%S)/
cp -r cache test-results/$(date +%Y%m%d-%H%M%S)/cache-sample
```

### Rollback (If Needed)
```bash
# Restore backups if testing corrupted state
[[ -d cache.backup-* ]] && rm -rf cache && mv cache.backup-* cache
[[ -f cost-tracking.jsonl.backup ]] && mv cost-tracking.jsonl.backup cost-tracking.jsonl
```

---

## Testing Results Template

After completing testing, document results:

```markdown
# Leonardo.ai Integration Testing Results

**Date**: YYYY-MM-DD
**Tester**: [Your Name/Persona]
**Environment**: [OS, shell version]
**API Key Status**: [Test mode / Real API key]

## Summary
- Total tests executed: X
- Passed: X
- Failed: X
- Skipped (no API key): X

## Issues Found
1. [Description of issue]
   - Severity: [Critical/Major/Minor]
   - Steps to reproduce: [...]
   - Expected: [...]
   - Actual: [...]

## Performance Metrics
- Average generation time: X seconds
- Total API cost: $X.XX
- Total images generated: X
- Total cache size: X MB

## Recommendations
- [ ] Ready for production use
- [ ] Needs fixes before production
- [ ] Documentation updates needed
- [ ] Additional testing needed in area: [...]
```

---

## Quick Reference Commands

**Test mode generation** (no API cost):
```bash
LEONARDO_API_KEY=test-mode ./generate-image.sh "Test prompt"
```

**Check total costs**:
```bash
jq -s 'map(.estimated_cost) | add' cost-tracking.jsonl
```

**Verify all images exist**:
```bash
jq -r '.image_path' generations.jsonl | while read p; do [[ -f "$p" ]] || echo "MISSING: $p"; done
```

**Clear cache and logs** (reset state):
```bash
rm -rf cache/*
rm -f cost-tracking.jsonl generations.jsonl
```

---

## Notes for Future Testing

- **API changes**: Periodically verify cost estimates against Leonardo.ai pricing
- **Model updates**: Test when Leonardo releases new models
- **Template evolution**: Add new templates and retest variation quality
- **Integration testing**: Test with actual novel chapter generation workflow
- **Monitoring setup**: After testing passes, set up cost alerts and error monitoring

**Document created**: 2025-11-21
**Last updated**: 2025-11-21
**Maintainer**: Experimenter persona (initial), Maintainer persona (ongoing)
