#!/usr/bin/env python3
"""
Day-by-day tavern simulation runner with incremental saves.
More resilient to API failures - saves after each day.
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tavern_simulation import TavernSimulation, TAVERN_PERSONAS

def run_day_by_day(seed=2026, output_prefix="tavern_incremental"):
    """Run simulation one day at a time with saves after each."""

    print("\n" + "="*70)
    print("TAVERN SIMULATION - DAY BY DAY MODE")
    print("="*70)

    sim = TavernSimulation(seed=seed)
    all_days = []

    for day_num in range(7):
        print(f"\n>>> Starting Day {day_num + 1}/7...")
        start = time.time()

        try:
            day_log = sim.simulate_day(day_num)
            all_days.append(day_log)

            # Save incremental progress
            save_incremental(all_days, sim, output_prefix, day_num + 1)

            elapsed = time.time() - start
            print(f">>> Day {day_num + 1} completed in {elapsed:.1f}s")

        except Exception as e:
            print(f">>> ERROR on Day {day_num + 1}: {e}")
            print(">>> Saving progress so far...")
            save_incremental(all_days, sim, output_prefix, day_num)
            raise

    # Final save with character summaries
    sim.week_log.days = all_days
    sim._generate_character_summaries()
    sim.save_results(output_prefix + "_final")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)

    return sim.week_log

def save_incremental(days, sim, prefix, day_count):
    """Save current progress to JSON."""

    def interaction_to_dict(i):
        return {
            "day": i.day,
            "time_of_day": i.time_of_day,
            "participants": i.participants,
            "context": i.context,
            "dialogue": i.dialogue,
            "mood": i.mood
        }

    def day_to_dict(d):
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
        "timestamp": datetime.now().isoformat(),
        "days_completed": day_count,
        "tavern_name": "The Rusty Anchor",
        "days": [day_to_dict(d) for d in days],
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

    filename = f"{prefix}_day{day_count}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"    [Saved progress to {filename}]")

if __name__ == "__main__":
    run_day_by_day(seed=2026)
