"""
Demographic Persona Generator for Club Harness.

Generates realistic agent personas with demographic attributes, personality traits,
and behavioral patterns for simulation and testing.

Inspired by repos/TinyTroupe/tinytroupe/persona_generator.py

Features:
- Big Five personality trait model
- Demographic distributions
- Behavioral tendency generation
- Persona consistency validation
- Persona presets for common use cases
"""

import random
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid


class BehaviorStyle(Enum):
    """Behavioral styles derived from personality."""
    ANALYTICAL = "analytical"     # High O, High C
    CREATIVE = "creative"         # High O, Low C
    SUPPORTIVE = "supportive"     # High A, High E
    ASSERTIVE = "assertive"       # Low A, High E
    CAUTIOUS = "cautious"         # High N, High C
    ADVENTUROUS = "adventurous"   # Low N, High O
    METHODICAL = "methodical"     # Low O, High C
    SPONTANEOUS = "spontaneous"   # Low C, High E


@dataclass
class BigFiveTraits:
    """
    Big Five personality traits model (OCEAN).

    Each trait is scored 0.0 to 1.0:
    - Low (0.0-0.33): Below average
    - Medium (0.33-0.67): Average
    - High (0.67-1.0): Above average
    """
    openness: float = 0.5        # Openness to experience (creativity, curiosity)
    conscientiousness: float = 0.5  # Organization, dependability
    extraversion: float = 0.5    # Sociability, assertiveness
    agreeableness: float = 0.5   # Cooperation, trust
    neuroticism: float = 0.5     # Emotional instability, anxiety

    def __post_init__(self):
        # Clamp all values to [0, 1]
        self.openness = max(0, min(1, self.openness))
        self.conscientiousness = max(0, min(1, self.conscientiousness))
        self.extraversion = max(0, min(1, self.extraversion))
        self.agreeableness = max(0, min(1, self.agreeableness))
        self.neuroticism = max(0, min(1, self.neuroticism))

    def get_dominant_traits(self) -> List[str]:
        """Get traits that are notably high or low."""
        dominant = []
        traits = {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
        }

        for name, value in traits.items():
            if value >= 0.7:
                dominant.append(f"high_{name}")
            elif value <= 0.3:
                dominant.append(f"low_{name}")

        return dominant

    def get_behavior_style(self) -> BehaviorStyle:
        """Derive behavioral style from traits."""
        # Map trait combinations to styles
        if self.openness > 0.6 and self.conscientiousness > 0.6:
            return BehaviorStyle.ANALYTICAL
        elif self.openness > 0.6 and self.conscientiousness < 0.4:
            return BehaviorStyle.CREATIVE
        elif self.agreeableness > 0.6 and self.extraversion > 0.6:
            return BehaviorStyle.SUPPORTIVE
        elif self.agreeableness < 0.4 and self.extraversion > 0.6:
            return BehaviorStyle.ASSERTIVE
        elif self.neuroticism > 0.6 and self.conscientiousness > 0.6:
            return BehaviorStyle.CAUTIOUS
        elif self.neuroticism < 0.4 and self.openness > 0.6:
            return BehaviorStyle.ADVENTUROUS
        elif self.openness < 0.4 and self.conscientiousness > 0.6:
            return BehaviorStyle.METHODICAL
        elif self.conscientiousness < 0.4 and self.extraversion > 0.6:
            return BehaviorStyle.SPONTANEOUS
        else:
            return BehaviorStyle.ANALYTICAL  # Default

    def to_dict(self) -> Dict[str, float]:
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BigFiveTraits":
        return cls(**data)

    def describe(self) -> str:
        """Generate natural language description of personality."""
        descriptions = []

        # Openness
        if self.openness > 0.7:
            descriptions.append("highly creative and curious")
        elif self.openness < 0.3:
            descriptions.append("practical and conventional")

        # Conscientiousness
        if self.conscientiousness > 0.7:
            descriptions.append("organized and reliable")
        elif self.conscientiousness < 0.3:
            descriptions.append("flexible and spontaneous")

        # Extraversion
        if self.extraversion > 0.7:
            descriptions.append("outgoing and energetic")
        elif self.extraversion < 0.3:
            descriptions.append("reserved and thoughtful")

        # Agreeableness
        if self.agreeableness > 0.7:
            descriptions.append("cooperative and trusting")
        elif self.agreeableness < 0.3:
            descriptions.append("competitive and skeptical")

        # Neuroticism
        if self.neuroticism > 0.7:
            descriptions.append("sensitive and anxious")
        elif self.neuroticism < 0.3:
            descriptions.append("calm and resilient")

        if descriptions:
            return ", ".join(descriptions)
        return "balanced personality"


@dataclass
class Demographics:
    """Demographic attributes for a persona."""
    age: int = 30
    gender: str = "neutral"
    occupation: str = "professional"
    education: str = "college"
    location: str = "urban"
    language: str = "english"
    cultural_background: str = "western"

    # Derived attributes
    experience_years: int = 5
    expertise_level: str = "intermediate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'age': self.age,
            'gender': self.gender,
            'occupation': self.occupation,
            'education': self.education,
            'location': self.location,
            'language': self.language,
            'cultural_background': self.cultural_background,
            'experience_years': self.experience_years,
            'expertise_level': self.expertise_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Demographics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Persona:
    """A complete agent persona."""
    persona_id: str
    name: str
    traits: BigFiveTraits
    demographics: Demographics

    # Identity
    role: str = "assistant"
    description: str = ""
    backstory: str = ""

    # Behavioral tendencies
    communication_style: str = "neutral"
    decision_style: str = "balanced"
    values: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)

    # Constraints
    constraints: List[str] = field(default_factory=list)
    avoid_topics: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'persona_id': self.persona_id,
            'name': self.name,
            'traits': self.traits.to_dict(),
            'demographics': self.demographics.to_dict(),
            'role': self.role,
            'description': self.description,
            'backstory': self.backstory,
            'communication_style': self.communication_style,
            'decision_style': self.decision_style,
            'values': self.values,
            'interests': self.interests,
            'constraints': self.constraints,
            'avoid_topics': self.avoid_topics,
            'created_at': self.created_at.isoformat(),
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        return cls(
            persona_id=data['persona_id'],
            name=data['name'],
            traits=BigFiveTraits.from_dict(data['traits']),
            demographics=Demographics.from_dict(data['demographics']),
            role=data.get('role', 'assistant'),
            description=data.get('description', ''),
            backstory=data.get('backstory', ''),
            communication_style=data.get('communication_style', 'neutral'),
            decision_style=data.get('decision_style', 'balanced'),
            values=data.get('values', []),
            interests=data.get('interests', []),
            constraints=data.get('constraints', []),
            avoid_topics=data.get('avoid_topics', []),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            version=data.get('version', '1.0'),
        )

    def to_system_prompt(self) -> str:
        """Generate a system prompt from the persona."""
        parts = [f"You are {self.name}, a {self.role}."]

        if self.description:
            parts.append(self.description)

        # Personality
        personality_desc = self.traits.describe()
        if personality_desc != "balanced personality":
            parts.append(f"Your personality is {personality_desc}.")

        # Demographics context
        demo = self.demographics
        if demo.occupation != "professional":
            parts.append(f"You work as a {demo.occupation}.")
        if demo.expertise_level != "intermediate":
            parts.append(f"You have {demo.expertise_level} expertise ({demo.experience_years} years of experience).")

        # Communication style
        style_map = {
            'formal': "Communicate formally and professionally.",
            'casual': "Communicate in a friendly, casual manner.",
            'technical': "Use precise technical language.",
            'empathetic': "Be warm, understanding, and empathetic.",
            'direct': "Be direct and to the point.",
        }
        if self.communication_style in style_map:
            parts.append(style_map[self.communication_style])

        # Values
        if self.values:
            parts.append(f"You value: {', '.join(self.values)}.")

        # Constraints
        if self.constraints:
            parts.append("Important guidelines: " + "; ".join(self.constraints))

        # Backstory (optional)
        if self.backstory:
            parts.append(f"Background: {self.backstory}")

        return " ".join(parts)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate persona consistency."""
        issues = []

        # Check trait values
        for trait_name, value in self.traits.to_dict().items():
            if not 0 <= value <= 1:
                issues.append(f"Trait {trait_name} out of range: {value}")

        # Check age consistency
        if self.demographics.age < 18 and self.demographics.expertise_level == "expert":
            issues.append("Age too young for expert expertise level")

        if self.demographics.experience_years > self.demographics.age - 16:
            issues.append("Experience years exceeds plausible working age")

        # Check name
        if not self.name or len(self.name) < 2:
            issues.append("Name is too short or missing")

        return len(issues) == 0, issues


class PersonaPresets:
    """Pre-built persona templates for common use cases."""

    @staticmethod
    def helpful_assistant() -> Persona:
        """Standard helpful assistant persona."""
        return Persona(
            persona_id=str(uuid.uuid4())[:12],
            name="Alex",
            traits=BigFiveTraits(
                openness=0.7,
                conscientiousness=0.8,
                extraversion=0.6,
                agreeableness=0.8,
                neuroticism=0.3,
            ),
            demographics=Demographics(
                age=30,
                occupation="assistant",
                expertise_level="advanced",
            ),
            role="helpful assistant",
            description="A knowledgeable and friendly assistant ready to help with any task.",
            communication_style="friendly",
            values=["helpfulness", "accuracy", "clarity"],
        )

    @staticmethod
    def technical_expert() -> Persona:
        """Technical expert persona for coding/engineering tasks."""
        return Persona(
            persona_id=str(uuid.uuid4())[:12],
            name="Dr. Chen",
            traits=BigFiveTraits(
                openness=0.8,
                conscientiousness=0.9,
                extraversion=0.4,
                agreeableness=0.6,
                neuroticism=0.3,
            ),
            demographics=Demographics(
                age=45,
                occupation="software engineer",
                education="doctorate",
                experience_years=20,
                expertise_level="expert",
            ),
            role="technical expert",
            description="A senior software engineer with deep expertise in systems design and architecture.",
            communication_style="technical",
            decision_style="analytical",
            values=["precision", "efficiency", "best practices"],
            interests=["distributed systems", "algorithms", "clean code"],
        )

    @staticmethod
    def creative_writer() -> Persona:
        """Creative writer persona for content generation."""
        return Persona(
            persona_id=str(uuid.uuid4())[:12],
            name="Jordan",
            traits=BigFiveTraits(
                openness=0.95,
                conscientiousness=0.5,
                extraversion=0.7,
                agreeableness=0.7,
                neuroticism=0.5,
            ),
            demographics=Demographics(
                age=35,
                occupation="writer",
                education="masters",
                experience_years=12,
                expertise_level="expert",
            ),
            role="creative writer",
            description="An imaginative writer with a flair for engaging storytelling.",
            communication_style="casual",
            decision_style="intuitive",
            values=["creativity", "authenticity", "engagement"],
            interests=["literature", "storytelling", "wordplay"],
        )

    @staticmethod
    def business_analyst() -> Persona:
        """Business analyst persona for data/strategy tasks."""
        return Persona(
            persona_id=str(uuid.uuid4())[:12],
            name="Morgan",
            traits=BigFiveTraits(
                openness=0.6,
                conscientiousness=0.85,
                extraversion=0.6,
                agreeableness=0.5,
                neuroticism=0.4,
            ),
            demographics=Demographics(
                age=38,
                occupation="business analyst",
                education="mba",
                experience_years=15,
                expertise_level="expert",
            ),
            role="business analyst",
            description="A strategic thinker who excels at data-driven decision making.",
            communication_style="formal",
            decision_style="analytical",
            values=["data-driven", "strategic", "results-oriented"],
            interests=["market analysis", "strategy", "optimization"],
        )

    @staticmethod
    def supportive_coach() -> Persona:
        """Supportive coach persona for guidance/mentoring."""
        return Persona(
            persona_id=str(uuid.uuid4())[:12],
            name="Sam",
            traits=BigFiveTraits(
                openness=0.7,
                conscientiousness=0.7,
                extraversion=0.8,
                agreeableness=0.9,
                neuroticism=0.2,
            ),
            demographics=Demographics(
                age=42,
                occupation="coach",
                education="masters",
                experience_years=18,
                expertise_level="expert",
            ),
            role="supportive coach",
            description="A warm and encouraging coach who helps others achieve their goals.",
            communication_style="empathetic",
            decision_style="collaborative",
            values=["growth", "encouragement", "potential"],
            interests=["personal development", "motivation", "goal-setting"],
        )


class PersonaGenerator:
    """
    Generates diverse, realistic personas.

    Supports:
    - Random generation with distributions
    - Constrained generation (specify some attributes)
    - Demographic-based generation
    - Personality-first generation
    """

    # Demographic distributions (simplified)
    OCCUPATIONS = [
        ("software engineer", 0.15),
        ("data scientist", 0.08),
        ("product manager", 0.08),
        ("designer", 0.06),
        ("researcher", 0.06),
        ("teacher", 0.08),
        ("consultant", 0.07),
        ("analyst", 0.08),
        ("writer", 0.05),
        ("healthcare professional", 0.06),
        ("entrepreneur", 0.05),
        ("student", 0.08),
        ("other professional", 0.10),
    ]

    EDUCATION_LEVELS = [
        ("high_school", 0.25),
        ("college", 0.35),
        ("bachelors", 0.20),
        ("masters", 0.12),
        ("doctorate", 0.05),
        ("professional", 0.03),
    ]

    NAMES = [
        "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
        "Quinn", "Avery", "Cameron", "Blake", "Drew", "Sage",
        "Rowan", "Skyler", "Reese", "Charlie", "Emery", "Dakota",
        "Jamie", "Phoenix", "River", "Eden", "Kai", "Finley",
    ]

    COMMUNICATION_STYLES = ["formal", "casual", "technical", "empathetic", "direct", "neutral"]
    DECISION_STYLES = ["analytical", "intuitive", "collaborative", "decisive", "balanced"]

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        self.rng = random.Random(seed)

    def generate(
        self,
        name: Optional[str] = None,
        traits: Optional[BigFiveTraits] = None,
        demographics: Optional[Demographics] = None,
        role: str = "assistant",
        **kwargs,
    ) -> Persona:
        """
        Generate a persona with optional constraints.

        Args:
            name: Specific name (random if None)
            traits: Specific traits (random if None)
            demographics: Specific demographics (random if None)
            role: Role for the persona
            **kwargs: Additional persona attributes

        Returns:
            Generated Persona
        """
        # Generate name
        if name is None:
            name = self.rng.choice(self.NAMES)

        # Generate traits
        if traits is None:
            traits = self._generate_traits()

        # Generate demographics
        if demographics is None:
            demographics = self._generate_demographics()

        # Derive communication style from traits
        communication_style = kwargs.get('communication_style')
        if communication_style is None:
            communication_style = self._derive_communication_style(traits)

        # Derive decision style from traits
        decision_style = kwargs.get('decision_style')
        if decision_style is None:
            decision_style = self._derive_decision_style(traits)

        # Generate values based on traits
        values = kwargs.get('values')
        if values is None:
            values = self._generate_values(traits)

        # Generate interests based on demographics
        interests = kwargs.get('interests')
        if interests is None:
            interests = self._generate_interests(demographics)

        # Generate description
        description = kwargs.get('description')
        if description is None:
            description = self._generate_description(name, role, traits, demographics)

        persona = Persona(
            persona_id=str(uuid.uuid4())[:12],
            name=name,
            traits=traits,
            demographics=demographics,
            role=role,
            description=description,
            communication_style=communication_style,
            decision_style=decision_style,
            values=values,
            interests=interests,
            constraints=kwargs.get('constraints', []),
            avoid_topics=kwargs.get('avoid_topics', []),
            backstory=kwargs.get('backstory', ''),
        )

        return persona

    def _generate_traits(self) -> BigFiveTraits:
        """Generate random Big Five traits with realistic distribution."""
        # Use beta distribution for more realistic trait distributions
        # Most people cluster around the middle
        def random_trait():
            # Beta(2, 2) gives a bell curve centered at 0.5
            return self.rng.betavariate(2, 2)

        return BigFiveTraits(
            openness=random_trait(),
            conscientiousness=random_trait(),
            extraversion=random_trait(),
            agreeableness=random_trait(),
            neuroticism=random_trait(),
        )

    def _generate_demographics(self) -> Demographics:
        """Generate random demographics."""
        # Age: weighted toward working age
        age = int(self.rng.gauss(35, 10))
        age = max(18, min(70, age))

        # Occupation: weighted selection
        occupation = self._weighted_choice(self.OCCUPATIONS)

        # Education: weighted selection
        education = self._weighted_choice(self.EDUCATION_LEVELS)

        # Experience based on age and education
        min_work_age = 18 if education == "high_school" else 22 if education in ["college", "bachelors"] else 26
        max_experience = max(0, age - min_work_age)
        experience_years = self.rng.randint(0, max_experience) if max_experience > 0 else 0

        # Expertise level based on experience
        if experience_years < 2:
            expertise_level = "beginner"
        elif experience_years < 5:
            expertise_level = "intermediate"
        elif experience_years < 10:
            expertise_level = "advanced"
        else:
            expertise_level = "expert"

        return Demographics(
            age=age,
            gender=self.rng.choice(["male", "female", "non-binary"]),
            occupation=occupation,
            education=education,
            location=self.rng.choice(["urban", "suburban", "rural"]),
            experience_years=experience_years,
            expertise_level=expertise_level,
        )

    def _weighted_choice(self, choices: List[Tuple[str, float]]) -> str:
        """Make a weighted random choice."""
        total = sum(w for _, w in choices)
        r = self.rng.random() * total
        cumulative = 0
        for choice, weight in choices:
            cumulative += weight
            if r <= cumulative:
                return choice
        return choices[-1][0]

    def _derive_communication_style(self, traits: BigFiveTraits) -> str:
        """Derive communication style from personality traits."""
        if traits.agreeableness > 0.7 and traits.extraversion > 0.6:
            return "empathetic"
        elif traits.conscientiousness > 0.7 and traits.openness < 0.4:
            return "formal"
        elif traits.extraversion > 0.7 and traits.agreeableness < 0.4:
            return "direct"
        elif traits.openness > 0.7:
            return "casual"
        elif traits.conscientiousness > 0.7:
            return "technical"
        return "neutral"

    def _derive_decision_style(self, traits: BigFiveTraits) -> str:
        """Derive decision style from personality traits."""
        if traits.conscientiousness > 0.7 and traits.openness > 0.6:
            return "analytical"
        elif traits.openness > 0.7 and traits.conscientiousness < 0.5:
            return "intuitive"
        elif traits.agreeableness > 0.7 and traits.extraversion > 0.6:
            return "collaborative"
        elif traits.extraversion > 0.7 and traits.neuroticism < 0.4:
            return "decisive"
        return "balanced"

    def _generate_values(self, traits: BigFiveTraits) -> List[str]:
        """Generate values based on personality."""
        values = []

        if traits.conscientiousness > 0.6:
            values.extend(["reliability", "precision", "organization"])
        if traits.openness > 0.6:
            values.extend(["creativity", "innovation", "learning"])
        if traits.agreeableness > 0.6:
            values.extend(["cooperation", "harmony", "empathy"])
        if traits.extraversion > 0.6:
            values.extend(["engagement", "connection", "enthusiasm"])
        if traits.neuroticism < 0.4:
            values.extend(["stability", "resilience", "calm"])

        # Pick 3-5 values
        self.rng.shuffle(values)
        return values[:self.rng.randint(3, min(5, len(values)))] if values else ["helpfulness"]

    def _generate_interests(self, demographics: Demographics) -> List[str]:
        """Generate interests based on demographics."""
        occupation_interests = {
            "software engineer": ["programming", "technology", "open source"],
            "data scientist": ["statistics", "machine learning", "data visualization"],
            "designer": ["aesthetics", "user experience", "visual arts"],
            "writer": ["literature", "storytelling", "language"],
            "researcher": ["academia", "discovery", "methodology"],
            "teacher": ["education", "mentoring", "learning"],
        }

        interests = occupation_interests.get(demographics.occupation, ["professional development"])

        # Add some random interests
        general_interests = [
            "reading", "music", "travel", "fitness", "cooking",
            "photography", "gaming", "movies", "nature", "science",
        ]
        interests.extend(self.rng.sample(general_interests, 2))

        return interests

    def _generate_description(
        self,
        name: str,
        role: str,
        traits: BigFiveTraits,
        demographics: Demographics,
    ) -> str:
        """Generate a natural description of the persona."""
        personality = traits.describe()

        parts = [
            f"A {demographics.expertise_level} {demographics.occupation}",
            f"who is {personality}",
        ]

        if demographics.experience_years > 10:
            parts.append(f"with {demographics.experience_years} years of experience")

        return " ".join(parts) + "."

    def generate_diverse_set(
        self,
        count: int = 5,
        ensure_diversity: bool = True,
    ) -> List[Persona]:
        """
        Generate a diverse set of personas.

        Args:
            count: Number of personas to generate
            ensure_diversity: Ensure variety in traits and demographics

        Returns:
            List of diverse Personas
        """
        personas = []
        used_names = set()
        used_occupations = set()

        for i in range(count):
            # Ensure name diversity
            available_names = [n for n in self.NAMES if n not in used_names]
            if not available_names:
                available_names = self.NAMES

            name = self.rng.choice(available_names)
            used_names.add(name)

            # Generate with some diversity constraints
            if ensure_diversity and i > 0:
                # Vary traits more
                traits = self._generate_traits()

                # Try for occupation diversity
                occupation = None
                for _ in range(5):
                    demo = self._generate_demographics()
                    if demo.occupation not in used_occupations:
                        occupation = demo.occupation
                        used_occupations.add(occupation)
                        break
                else:
                    demo = self._generate_demographics()
            else:
                traits = None
                demo = None

            persona = self.generate(name=name, traits=traits, demographics=demo if demo else None)
            personas.append(persona)

        return personas
