#!/usr/bin/env python3
"""
The Rusty Anchor Tavern - A Week-Long AI Simulation

A living tavern simulation with persistent personalities who interact
over the course of a week, building relationships, sharing stories,
and creating emergent narratives.

Uses Club Harness with OpenRouter for LLM-powered character interactions.
"""

import os
import sys
import json
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add club_harness to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.core.agent import Agent, AgentBuilder
from club_harness.personas import BigFiveTraits, Persona
from club_harness.llm.router import LLMRouter


# ============================================================================
# TAVERN PERSONAS
# ============================================================================

TAVERN_PERSONAS = {
    "Grom": {
        "role": "Bartender & Owner",
        "age": 52,
        "description": "A barrel-chested former soldier who bought the Rusty Anchor with his military pension. His scarred hands tell stories he rarely shares. Gruff exterior hides a protective nature toward his regulars.",
        "traits": BigFiveTraits(
            openness=0.4,        # Traditional, practical
            conscientiousness=0.85,  # Highly responsible
            extraversion=0.35,   # Reserved but commanding
            agreeableness=0.5,   # Fair but firm
            neuroticism=0.25,    # Steady under pressure
        ),
        "values": ["loyalty", "honesty", "hard work"],
        "speech_style": "Speaks in short, direct sentences. Uses old soldier expressions. Rarely asks questions - makes statements. Occasionally grunts as acknowledgment.",
        "backstory": "Served 20 years in the King's infantry. Lost his squad in the Battle of Thornhill. The tavern is his way of building a new family.",
        "relationships": {
            "Mira": "protective, like a daughter",
            "Dice": "distrustful but tolerant - good for business",
            "Old Theo": "deep respect, fellow veteran",
            "Merchant Pell": "useful source of news, keeps him close",
        }
    },
    "Mira": {
        "role": "Waitress",
        "age": 23,
        "description": "Quick-witted and observant, Mira sees everything that happens in the tavern. She dreams of saving enough to open her own tea house someday. Her infectious laugh can lighten any mood.",
        "traits": BigFiveTraits(
            openness=0.8,        # Creative, curious
            conscientiousness=0.7,   # Hardworking
            extraversion=0.85,   # Very social
            agreeableness=0.9,   # Warm and kind
            neuroticism=0.45,    # Occasionally anxious about future
        ),
        "values": ["kindness", "dreams", "family"],
        "speech_style": "Warm and chatty. Uses endearments like 'love' and 'dear'. Asks lots of questions. Laughs easily. Sometimes trails off when thinking about the future.",
        "backstory": "Orphaned young, taken in by Grom three years ago. The tavern is the only real home she's known.",
        "relationships": {
            "Grom": "sees him as a father figure",
            "Dice": "finds him charming but worries about him",
            "Old Theo": "loves his stories, like a grandfather",
            "Merchant Pell": "good tipper, but something feels off about him",
        }
    },
    "Dice": {
        "role": "Professional Gambler",
        "age": 34,
        "description": "Handsome and silver-tongued, Dice makes his living reading people and playing odds. Always impeccably dressed despite uncertain income. Has a mysterious past he deflects with humor.",
        "traits": BigFiveTraits(
            openness=0.7,        # Adaptable, quick-thinking
            conscientiousness=0.4,   # Lives in the moment
            extraversion=0.8,    # Charming, social
            agreeableness=0.55,  # Can be manipulative but not cruel
            neuroticism=0.5,     # Hidden anxieties about debts
        ),
        "values": ["freedom", "wit", "luck"],
        "speech_style": "Smooth and confident. Uses gambling metaphors. Deflects personal questions with jokes. Compliments strategically. Voice drops when serious.",
        "backstory": "Son of a minor noble who lost everything. Learned cards to survive, now can't stop. Running from debts in three cities.",
        "relationships": {
            "Grom": "respects him, knows not to cross him",
            "Mira": "genuinely fond of her, protective",
            "Old Theo": "enjoys the stories, reminds him of better times",
            "Merchant Pell": "mutual wariness - both recognize a schemer",
        }
    },
    "Old Theo": {
        "role": "Retired Sailor & Storyteller",
        "age": 71,
        "description": "Weathered and wise, Old Theo has sailed to lands most folk think are legends. His stories might be half exaggeration, but the kernel of truth in each one is stranger than fiction.",
        "traits": BigFiveTraits(
            openness=0.9,        # Full of wonder
            conscientiousness=0.5,   # Relaxed about details
            extraversion=0.7,    # Loves an audience
            agreeableness=0.85,  # Gentle soul
            neuroticism=0.3,     # At peace with life
        ),
        "values": ["adventure", "truth hidden in tales", "legacy"],
        "speech_style": "Rambling, colorful storyteller. Uses nautical terms. Says 'Now, the thing of it is...' Often gets sidetracked. Pauses dramatically.",
        "backstory": "Sailed the Sapphire Seas for 50 years. Wife passed ten years ago. The tavern is where he feels useful.",
        "relationships": {
            "Grom": "fellow veteran, unspoken understanding",
            "Mira": "she reminds him of his granddaughter",
            "Dice": "sees through the charm to the scared boy",
            "Merchant Pell": "suspicious of his stories",
        }
    },
    "Merchant Pell": {
        "role": "Traveling Merchant & Gossip",
        "age": 45,
        "description": "A portly, well-connected trader who knows everyone's business. His friendly demeanor masks a calculating mind. Always has exotic goods and fresher news to sell.",
        "traits": BigFiveTraits(
            openness=0.6,        # Curious about useful things
            conscientiousness=0.75,  # Organized, calculating
            extraversion=0.75,   # Networking is his trade
            agreeableness=0.45,  # Friendly facade, self-interested
            neuroticism=0.6,     # Worried about rivals
        ),
        "values": ["profit", "information", "connections"],
        "speech_style": "Effusive and gossipy. Drops names constantly. Says 'between you and me' often. Asks probing questions casually. Laughs too readily.",
        "backstory": "Built his network from nothing. Has information on half the town's secrets. Uses the tavern to gather and spread news strategically.",
        "relationships": {
            "Grom": "keeps on good terms - dangerous to cross",
            "Mira": "tips well to keep her talking",
            "Dice": "potential mark or competition",
            "Old Theo": "his stories sometimes contain valuable info",
        }
    }
}

# Transient visitors who pass through
TRANSIENT_POOL = [
    {
        "name": "Brother Marcus",
        "role": "Traveling Monk",
        "description": "A contemplative monk on pilgrimage. Speaks little but listens much.",
        "speech_style": "Soft-spoken, asks philosophical questions, offers blessings.",
    },
    {
        "name": "Sergeant Voss",
        "role": "Military Recruiter",
        "description": "A stern military officer looking for able bodies for the northern campaign.",
        "speech_style": "Clipped military speech, evaluates everyone, mentions good pay.",
    },
    {
        "name": "Lyra the Bard",
        "role": "Traveling Musician",
        "description": "A cheerful bard with a lute, trading songs for room and board.",
        "speech_style": "Musical, quotes poetry, speaks in metaphors, always optimistic.",
    },
    {
        "name": "Garret Stone",
        "role": "Bounty Hunter",
        "description": "A quiet, dangerous-looking man asking careful questions about someone.",
        "speech_style": "Few words, watches everyone, asks about specific people.",
    },
    {
        "name": "Widow Marsh",
        "role": "Herbalist",
        "description": "An elderly woman selling remedies and poultices, knows folk medicine.",
        "speech_style": "Grandmotherly, offers health advice, speaks of old remedies.",
    },
    {
        "name": "Young Tom",
        "role": "Farm Boy",
        "description": "A wide-eyed teenager from a village, first time in a real tavern.",
        "speech_style": "Nervous, easily impressed, asks naive questions about city life.",
    },
]

# Daily events that can trigger interactions
DAILY_EVENTS = {
    "Moonday": [
        "The weekly card game begins in the corner booth",
        "A merchant caravan arrives in town",
        "Rain drives everyone indoors early",
    ],
    "Twosday": [
        "Market day brings a full house",
        "A traveling performer asks to play",
        "Someone's purse goes missing",
    ],
    "Thirdsday": [
        "A brawl nearly breaks out over an old grudge",
        "A stranger asks too many questions",
        "Old Theo's birthday celebration",
    ],
    "Fourthday": [
        "The local guard captain visits",
        "A heavy storm keeps everyone trapped inside",
        "News arrives from the capital",
    ],
    "Fifthday": [
        "The weekly dice tournament",
        "A mysterious package arrives for someone",
        "A former patron returns after years away",
    ],
    "Sixthday": [
        "The busiest night of the week",
        "Live music and dancing",
        "A proposal happens in the tavern",
    ],
    "Seventhday": [
        "A quiet, reflective day",
        "Only regulars at the bar",
        "Grom shares a rare story",
    ],
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Interaction:
    """A single interaction between characters."""
    day: str
    time_of_day: str  # morning, afternoon, evening, night
    participants: list[str]
    context: str
    dialogue: list[dict]  # [{"speaker": "...", "text": "..."}, ...]
    mood: str  # jovial, tense, melancholy, mysterious, etc.

@dataclass
class DayLog:
    """All events and interactions for a single day."""
    day_name: str
    day_number: int
    weather: str
    events: list[str]
    transients: list[str]
    interactions: list[Interaction]
    notable_moments: list[str] = field(default_factory=list)

@dataclass
class TavernWeek:
    """The complete week's simulation."""
    start_date: str
    tavern_name: str = "The Rusty Anchor"
    days: list[DayLog] = field(default_factory=list)
    character_arcs: dict = field(default_factory=dict)


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class TavernSimulation:
    """Main simulation engine for the tavern."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.router = LLMRouter()
        self.agents: dict[str, Agent] = {}
        self.week_log = TavernWeek(start_date=datetime.now().isoformat())
        self.running_narratives = []  # Story threads that develop

        # Initialize agents for all regular characters
        self._initialize_agents()

    def _chat_with_retry(self, agent: Agent, message: str, max_retries: int = 4) -> str:
        """Chat with exponential backoff retry for API errors."""
        for attempt in range(max_retries):
            try:
                return agent.chat(message)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4, 8 seconds
                    print(f"    [API retry {attempt + 1}/{max_retries}, waiting {wait_time}s...]")
                    time.sleep(wait_time)
                else:
                    # Return a fallback response
                    return f"*{agent.name} nods quietly, lost in thought*"

    def _initialize_agents(self):
        """Create AI agents for each tavern character."""
        print("\n[Initializing Tavern Characters]")

        for name, persona_data in TAVERN_PERSONAS.items():
            system_prompt = self._build_character_prompt(name, persona_data)

            agent = (
                AgentBuilder(name)
                .with_instructions(system_prompt)
                .with_tier("free")  # Use free models
                .build()
            )

            self.agents[name] = agent
            print(f"  - {name}: {persona_data['role']}")

    def _build_character_prompt(self, name: str, persona: dict) -> str:
        """Build a detailed system prompt for a character."""
        relationships = "\n".join(
            f"  - {other}: {rel}"
            for other, rel in persona.get("relationships", {}).items()
        )

        return f"""You are {name}, {persona['role']} at The Rusty Anchor tavern.

ABOUT YOU:
{persona['description']}

Age: {persona['age']}
Values: {', '.join(persona['values'])}
Backstory: {persona['backstory']}

HOW YOU SPEAK:
{persona['speech_style']}

YOUR RELATIONSHIPS:
{relationships}

ROLEPLAY INSTRUCTIONS:
- Stay completely in character as {name}
- Respond naturally to the situation and other characters
- Draw on your backstory and relationships
- Keep responses conversational (2-4 sentences typically)
- React authentically based on your personality traits
- Never break character or mention being an AI
- Reference past events when relevant
- Show your personality through word choice and mannerisms"""

    def _create_transient_agent(self, transient: dict) -> Agent:
        """Create a temporary agent for a transient visitor."""
        system_prompt = f"""You are {transient['name']}, a {transient['role']} passing through The Rusty Anchor tavern.

{transient['description']}

How you speak: {transient['speech_style']}

Stay in character. Keep responses brief (1-3 sentences). You're just passing through."""

        return (
            AgentBuilder(transient['name'])
            .with_instructions(system_prompt)
            .with_tier("free")
            .build()
        )

    def _generate_interaction(
        self,
        participants: list[str],
        context: str,
        day_name: str,
        time_of_day: str,
        mood: str = "casual",
        num_exchanges: int = 4
    ) -> Interaction:
        """Generate a multi-turn interaction between characters."""

        dialogue = []

        # Build context message
        scene_setting = f"""[Scene: The Rusty Anchor Tavern, {day_name} {time_of_day}]
[Context: {context}]
[Mood: {mood}]
[Present: {', '.join(participants)}]

Respond naturally to this scene. What do you say or do?"""

        # Get first speaker's opening
        first_speaker = participants[0]
        agent = self.agents.get(first_speaker)

        if not agent:
            # Must be a transient
            transient_data = next(
                (t for t in TRANSIENT_POOL if t['name'] == first_speaker),
                None
            )
            if transient_data:
                agent = self._create_transient_agent(transient_data)
                self.agents[first_speaker] = agent

        if agent:
            response = self._chat_with_retry(agent, scene_setting)
            dialogue.append({"speaker": first_speaker, "text": response})

        # Continue the conversation
        for i in range(num_exchanges - 1):
            # Pick next speaker (rotate through participants)
            speaker_idx = (i + 1) % len(participants)
            speaker = participants[speaker_idx]

            agent = self.agents.get(speaker)
            if not agent:
                transient_data = next(
                    (t for t in TRANSIENT_POOL if t['name'] == speaker),
                    None
                )
                if transient_data:
                    agent = self._create_transient_agent(transient_data)
                    self.agents[speaker] = agent

            if agent:
                # Build context from previous dialogue
                recent = dialogue[-2:] if len(dialogue) >= 2 else dialogue
                context_msg = "\n".join(
                    f"{d['speaker']}: \"{d['text']}\"" for d in recent
                )

                prompt = f"""[Continuing scene at The Rusty Anchor]
Recent conversation:
{context_msg}

How do you respond as {speaker}?"""

                response = self._chat_with_retry(agent, prompt)
                dialogue.append({"speaker": speaker, "text": response})

        return Interaction(
            day=day_name,
            time_of_day=time_of_day,
            participants=participants,
            context=context,
            dialogue=dialogue,
            mood=mood
        )

    def simulate_day(self, day_number: int) -> DayLog:
        """Simulate a single day at the tavern."""

        day_names = ["Moonday", "Twosday", "Thirdsday", "Fourthday",
                     "Fifthday", "Sixthday", "Seventhday"]
        day_name = day_names[day_number % 7]

        weathers = ["Clear skies", "Overcast", "Light rain", "Heavy rain",
                    "Foggy morning", "Blustery winds", "Perfect weather"]
        weather = self.rng.choice(weathers)

        print(f"\n{'='*60}")
        print(f"DAY {day_number + 1}: {day_name}")
        print(f"Weather: {weather}")
        print(f"{'='*60}")

        # Select today's events
        possible_events = DAILY_EVENTS.get(day_name, ["A quiet day"])
        events = self.rng.sample(possible_events, min(2, len(possible_events)))

        # Decide which transients visit today
        num_transients = self.rng.randint(0, 2)
        transients = self.rng.sample(TRANSIENT_POOL, num_transients)
        transient_names = [t['name'] for t in transients]

        print(f"Events: {events}")
        print(f"Visitors: {transient_names if transient_names else 'None'}")

        interactions = []

        # Morning interaction (if appropriate)
        if self.rng.random() > 0.4:
            print("\n[Morning Scene]")
            morning_participants = self.rng.sample(
                ["Grom", "Mira", "Old Theo"],
                k=self.rng.randint(2, 3)
            )
            morning_contexts = [
                "Opening up the tavern, preparing for the day",
                "Quiet morning coffee before customers arrive",
                "Discussing yesterday's events",
            ]
            interaction = self._generate_interaction(
                participants=morning_participants,
                context=self.rng.choice(morning_contexts),
                day_name=day_name,
                time_of_day="morning",
                mood="peaceful",
                num_exchanges=3
            )
            interactions.append(interaction)
            self._print_interaction(interaction)

        # Afternoon interaction
        print("\n[Afternoon Scene]")
        afternoon_participants = ["Mira"]
        afternoon_participants.extend(
            self.rng.sample(
                ["Dice", "Merchant Pell", "Old Theo"],
                k=self.rng.randint(1, 2)
            )
        )
        if transient_names and self.rng.random() > 0.5:
            afternoon_participants.append(self.rng.choice(transient_names))

        interaction = self._generate_interaction(
            participants=afternoon_participants,
            context=events[0] if events else "Regular afternoon at the tavern",
            day_name=day_name,
            time_of_day="afternoon",
            mood=self.rng.choice(["casual", "lively", "curious"]),
            num_exchanges=4
        )
        interactions.append(interaction)
        self._print_interaction(interaction)

        # Evening interaction (main event)
        print("\n[Evening Scene]")
        evening_participants = ["Grom", "Dice", "Old Theo"]
        if "Merchant Pell" not in afternoon_participants or self.rng.random() > 0.5:
            evening_participants.append("Merchant Pell")
        if transient_names:
            evening_participants.extend(
                [t for t in transient_names if t not in afternoon_participants][:1]
            )

        evening_context = events[1] if len(events) > 1 else events[0] if events else "Busy evening at the tavern"

        interaction = self._generate_interaction(
            participants=evening_participants,
            context=evening_context,
            day_name=day_name,
            time_of_day="evening",
            mood=self.rng.choice(["jovial", "tense", "mysterious", "nostalgic"]),
            num_exchanges=5
        )
        interactions.append(interaction)
        self._print_interaction(interaction)

        # Late night (quieter, more intimate)
        if self.rng.random() > 0.5:
            print("\n[Late Night Scene]")
            night_participants = self.rng.sample(
                ["Grom", "Dice", "Old Theo", "Mira"],
                k=2
            )
            night_contexts = [
                "Closing time, only the dedicated remain",
                "A quiet moment between old friends",
                "Sharing secrets over the last drink",
                "Reflecting on the day's events",
            ]
            interaction = self._generate_interaction(
                participants=night_participants,
                context=self.rng.choice(night_contexts),
                day_name=day_name,
                time_of_day="late night",
                mood=self.rng.choice(["melancholy", "intimate", "contemplative"]),
                num_exchanges=3
            )
            interactions.append(interaction)
            self._print_interaction(interaction)

        return DayLog(
            day_name=day_name,
            day_number=day_number + 1,
            weather=weather,
            events=events,
            transients=transient_names,
            interactions=interactions
        )

    def _print_interaction(self, interaction: Interaction):
        """Pretty-print an interaction."""
        print(f"\n  Context: {interaction.context}")
        print(f"  Mood: {interaction.mood}")
        print(f"  ---")
        for entry in interaction.dialogue:
            speaker = entry['speaker']
            text = entry['text']
            # Truncate very long responses
            if len(text) > 300:
                text = text[:297] + "..."
            print(f"  {speaker}: \"{text}\"")

    def run_week(self) -> TavernWeek:
        """Run a full week simulation."""

        print("\n" + "="*70)
        print("THE RUSTY ANCHOR TAVERN")
        print("A Week in the Life - AI Simulation")
        print("="*70)

        for day_num in range(7):
            day_log = self.simulate_day(day_num)
            self.week_log.days.append(day_log)

        # Generate summary of character arcs
        self._generate_character_summaries()

        return self.week_log

    def _generate_character_summaries(self):
        """Generate brief summaries of each character's week."""
        print("\n" + "="*70)
        print("CHARACTER SUMMARIES")
        print("="*70)

        for name in TAVERN_PERSONAS.keys():
            # Count interactions
            interaction_count = sum(
                1 for day in self.week_log.days
                for interaction in day.interactions
                if name in interaction.participants
            )
            self.week_log.character_arcs[name] = {
                "interactions": interaction_count,
                "role": TAVERN_PERSONAS[name]["role"]
            }
            print(f"\n{name} ({TAVERN_PERSONAS[name]['role']})")
            print(f"  Appeared in {interaction_count} scenes")

    def save_results(self, filename: str = "tavern_week_results"):
        """Save the simulation results."""

        # Convert to serializable format
        def interaction_to_dict(i: Interaction) -> dict:
            return {
                "day": i.day,
                "time_of_day": i.time_of_day,
                "participants": i.participants,
                "context": i.context,
                "dialogue": i.dialogue,
                "mood": i.mood
            }

        def day_to_dict(d: DayLog) -> dict:
            return {
                "day_name": d.day_name,
                "day_number": d.day_number,
                "weather": d.weather,
                "events": d.events,
                "transients": d.transients,
                "interactions": [interaction_to_dict(i) for i in d.interactions],
                "notable_moments": d.notable_moments
            }

        output = {
            "start_date": self.week_log.start_date,
            "tavern_name": self.week_log.tavern_name,
            "days": [day_to_dict(d) for d in self.week_log.days],
            "character_arcs": self.week_log.character_arcs,
            "personas": {
                name: {
                    "role": p["role"],
                    "age": p["age"],
                    "description": p["description"],
                    "values": p["values"]
                }
                for name, p in TAVERN_PERSONAS.items()
            }
        }

        # Save JSON
        json_path = f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved JSON results to: {json_path}")

        # Save readable narrative
        narrative_path = f"{filename}.txt"
        with open(narrative_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("THE RUSTY ANCHOR TAVERN\n")
            f.write("A Week in the Life\n")
            f.write("="*70 + "\n\n")

            f.write("CAST OF CHARACTERS\n")
            f.write("-"*40 + "\n")
            for name, p in TAVERN_PERSONAS.items():
                f.write(f"\n{name} - {p['role']} (Age {p['age']})\n")
                f.write(f"  {p['description']}\n")
            f.write("\n")

            for day in self.week_log.days:
                f.write("\n" + "="*70 + "\n")
                f.write(f"DAY {day.day_number}: {day.day_name}\n")
                f.write(f"Weather: {day.weather}\n")
                f.write("="*70 + "\n")

                if day.events:
                    f.write(f"\nEvents: {', '.join(day.events)}\n")
                if day.transients:
                    f.write(f"Visitors: {', '.join(day.transients)}\n")

                for interaction in day.interactions:
                    f.write(f"\n--- {interaction.time_of_day.upper()} ---\n")
                    f.write(f"[{interaction.context}]\n")
                    f.write(f"Mood: {interaction.mood}\n\n")

                    for entry in interaction.dialogue:
                        f.write(f'{entry["speaker"]}: "{entry["text"]}"\n\n')

        print(f"Saved narrative to: {narrative_path}")

        return json_path, narrative_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the tavern simulation."""

    print("\n" + "*"*70)
    print("*" + " "*68 + "*")
    print("*" + "  THE RUSTY ANCHOR TAVERN SIMULATION".center(68) + "*")
    print("*" + "  Powered by Club Harness & OpenRouter".center(68) + "*")
    print("*" + " "*68 + "*")
    print("*"*70)

    # Create and run simulation
    sim = TavernSimulation(seed=42)
    week_log = sim.run_week()

    # Save results
    json_path, narrative_path = sim.save_results("tavern_week_results")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {narrative_path}")
    print(f"\nTotal days simulated: {len(week_log.days)}")
    total_interactions = sum(len(d.interactions) for d in week_log.days)
    print(f"Total interactions: {total_interactions}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
