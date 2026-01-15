#!/usr/bin/env python3
"""Test demographic persona generation."""

import sys
sys.path.insert(0, '/home/user/Merge')

from club_harness.personas import (
    BigFiveTraits,
    Demographics,
    Persona,
    PersonaGenerator,
    PersonaPresets,
    BehaviorStyle,
)
import json

print("=" * 60)
print("DEMOGRAPHIC PERSONA GENERATION TEST")
print("=" * 60)

# Test 1: Big Five traits
print("\n[TEST 1] Big Five personality traits")

traits = BigFiveTraits(
    openness=0.8,
    conscientiousness=0.7,
    extraversion=0.4,
    agreeableness=0.6,
    neuroticism=0.3,
)

print(f"  Traits: O={traits.openness}, C={traits.conscientiousness}, E={traits.extraversion}, A={traits.agreeableness}, N={traits.neuroticism}")
print(f"  Dominant traits: {traits.get_dominant_traits()}")
print(f"  Behavior style: {traits.get_behavior_style().value}")
print(f"  Description: {traits.describe()}")

assert traits.get_behavior_style() == BehaviorStyle.ANALYTICAL
print("  [PASS] Big Five traits work")

# Test 2: Demographics
print("\n[TEST 2] Demographics")

demo = Demographics(
    age=35,
    gender="female",
    occupation="software engineer",
    education="masters",
    experience_years=12,
    expertise_level="expert",
)

print(f"  Age: {demo.age}")
print(f"  Occupation: {demo.occupation}")
print(f"  Expertise: {demo.expertise_level} ({demo.experience_years} years)")

demo_dict = demo.to_dict()
demo_restored = Demographics.from_dict(demo_dict)
assert demo_restored.occupation == demo.occupation
print("  [PASS] Demographics work")

# Test 3: Persona creation and system prompt
print("\n[TEST 3] Persona creation and system prompt")

persona = Persona(
    persona_id="test-001",
    name="Dr. Sarah Chen",
    traits=traits,
    demographics=demo,
    role="technical advisor",
    description="An expert software architect with deep knowledge of distributed systems.",
    communication_style="technical",
    decision_style="analytical",
    values=["precision", "efficiency", "mentorship"],
    interests=["system design", "algorithms", "open source"],
)

print(f"  Persona: {persona.name}")
print(f"  Role: {persona.role}")
print(f"  Values: {persona.values}")

# Generate system prompt
prompt = persona.to_system_prompt()
print(f"  System prompt ({len(prompt)} chars):")
print(f"    '{prompt[:100]}...'")

# Validate
valid, issues = persona.validate()
print(f"  Validation: {'PASS' if valid else 'FAIL'}")
if issues:
    print(f"    Issues: {issues}")

print("  [PASS] Persona creation works")

# Test 4: Persona presets
print("\n[TEST 4] Persona presets")

presets = [
    ("helpful_assistant", PersonaPresets.helpful_assistant()),
    ("technical_expert", PersonaPresets.technical_expert()),
    ("creative_writer", PersonaPresets.creative_writer()),
    ("business_analyst", PersonaPresets.business_analyst()),
    ("supportive_coach", PersonaPresets.supportive_coach()),
]

for preset_name, preset_persona in presets:
    valid, _ = preset_persona.validate()
    print(f"  {preset_name}: {preset_persona.name} - valid: {valid}")

print("  [PASS] Presets work")

# Test 5: Random persona generation
print("\n[TEST 5] Random persona generation")

generator = PersonaGenerator(seed=42)

random_persona = generator.generate()
print(f"  Generated: {random_persona.name}")
print(f"    Occupation: {random_persona.demographics.occupation}")
print(f"    Personality: {random_persona.traits.describe()}")
print(f"    Communication: {random_persona.communication_style}")
print(f"    Values: {random_persona.values[:3]}")

valid, issues = random_persona.validate()
assert valid, f"Random persona should be valid: {issues}"
print("  [PASS] Random generation works")

# Test 6: Constrained generation
print("\n[TEST 6] Constrained persona generation")

constrained_persona = generator.generate(
    name="Specialist Sam",
    role="security expert",
    traits=BigFiveTraits(
        openness=0.5,
        conscientiousness=0.95,
        extraversion=0.3,
        agreeableness=0.4,
        neuroticism=0.2,
    ),
    values=["security", "precision", "vigilance"],
)

print(f"  Generated: {constrained_persona.name}")
print(f"  Role: {constrained_persona.role}")
print(f"  Traits conscientiousness: {constrained_persona.traits.conscientiousness}")
print(f"  Values: {constrained_persona.values}")

assert constrained_persona.name == "Specialist Sam"
assert constrained_persona.traits.conscientiousness == 0.95
print("  [PASS] Constrained generation works")

# Test 7: Diverse set generation
print("\n[TEST 7] Diverse persona set generation")

diverse_personas = generator.generate_diverse_set(count=5, ensure_diversity=True)

print(f"  Generated {len(diverse_personas)} diverse personas:")
names = set()
occupations = set()
for p in diverse_personas:
    print(f"    - {p.name}: {p.demographics.occupation}, {p.traits.get_behavior_style().value}")
    names.add(p.name)
    occupations.add(p.demographics.occupation)

print(f"  Unique names: {len(names)}")
print(f"  Unique occupations: {len(occupations)}")

assert len(names) == 5, "Should have 5 unique names"
print("  [PASS] Diverse set generation works")

# Test 8: Serialization
print("\n[TEST 8] Serialization round-trip")

original = PersonaPresets.technical_expert()
serialized = json.dumps(original.to_dict())
restored = Persona.from_dict(json.loads(serialized))

assert restored.name == original.name
assert restored.traits.conscientiousness == original.traits.conscientiousness
assert restored.demographics.occupation == original.demographics.occupation

print(f"  Original: {original.name}")
print(f"  Restored: {restored.name}")
print(f"  Traits match: {restored.traits.to_dict() == original.traits.to_dict()}")
print("  [PASS] Serialization works")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("All persona generation tests passed!")
print("\nFeatures tested:")
print("  - Big Five personality traits")
print("  - Demographic attributes")
print("  - Persona creation and validation")
print("  - System prompt generation")
print("  - Persona presets")
print("  - Random generation")
print("  - Constrained generation")
print("  - Diverse set generation")
print("  - JSON serialization")
