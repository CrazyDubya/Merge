#!/usr/bin/env python3
"""
Claude Daemon Throttle Control TUI
Interactive interface for adjusting daemon resource usage with presets and fine-tuning
"""

import curses
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DAEMON_ROOT = Path.home() / ".claude" / "daemon"
CONFIG_FILE = DAEMON_ROOT / "throttle-config.json"
PRESETS_FILE = DAEMON_ROOT / "throttle-presets.json"

# Default presets
DEFAULT_PRESETS = {
    "baseline": {
        "name": "Baseline (0%)",
        "description": "Original settings - no throttling",
        "reduction": 0,
        "settings": {
            "min_sleep": 600,
            "default_sleep": 900,
            "max_sleep": 1800,
            "night_sleep": 28800,
            "active_start": 7,
            "active_end": 22,
            "task_weight": 0.5,
            "reflection_weight": 0.3,
            "conversation_weight": 0.2,
            "reflection_cooldown": 60,
            "chaos_enabled": True,
            "chaos_probability": 0.10,
            "thinking_levels": {
                "experimenter": "think",
                "optimizer": "think",
                "maintainer": "think",
                "architect": "think hard",
                "skeptic": "think hard",
                "auditor": "think harder"
            }
        }
    },
    "light": {
        "name": "Light Throttle (15%)",
        "description": "Sleep +25%, Reflection 90min, Chaos 5%",
        "reduction": 15,
        "settings": {
            "min_sleep": 750,
            "default_sleep": 1125,
            "max_sleep": 2250,
            "night_sleep": 28800,
            "active_start": 7,
            "active_end": 22,
            "task_weight": 0.5,
            "reflection_weight": 0.3,
            "conversation_weight": 0.2,
            "reflection_cooldown": 90,
            "chaos_enabled": True,
            "chaos_probability": 0.05,
            "thinking_levels": {
                "experimenter": "think",
                "optimizer": "think",
                "maintainer": "think",
                "architect": "think hard",
                "skeptic": "think hard",
                "auditor": "think harder"
            }
        }
    },
    "medium": {
        "name": "Medium Throttle (35%)",
        "description": "Sleep +50%, 9AM-9PM, Reflection 120min, No chaos, Standard thinking",
        "reduction": 35,
        "settings": {
            "min_sleep": 900,
            "default_sleep": 1350,
            "max_sleep": 2700,
            "night_sleep": 28800,
            "active_start": 9,
            "active_end": 21,
            "task_weight": 0.5,
            "reflection_weight": 0.3,
            "conversation_weight": 0.2,
            "reflection_cooldown": 120,
            "chaos_enabled": False,
            "chaos_probability": 0.00,
            "thinking_levels": {
                "experimenter": "think",
                "optimizer": "think",
                "maintainer": "think",
                "architect": "think",
                "skeptic": "think",
                "auditor": "think hard"
            }
        }
    },
    "heavy": {
        "name": "Heavy Throttle (55%)",
        "description": "Sleep 2x, 9AM-6PM, Task-focused, Reflection 180min, No chaos, Minimal thinking",
        "reduction": 55,
        "settings": {
            "min_sleep": 1200,
            "default_sleep": 1800,
            "max_sleep": 3600,
            "night_sleep": 28800,
            "active_start": 9,
            "active_end": 18,
            "task_weight": 0.7,
            "reflection_weight": 0.15,
            "conversation_weight": 0.15,
            "reflection_cooldown": 180,
            "chaos_enabled": False,
            "chaos_probability": 0.00,
            "thinking_levels": {
                "experimenter": "think",
                "optimizer": "think",
                "maintainer": "think",
                "architect": "think",
                "skeptic": "think",
                "auditor": "think"
            }
        }
    }
}


class ThrottleConfig:
    """Manages throttle configuration"""

    def __init__(self):
        self.current = self.load_current()
        self.presets = self.load_presets()

    def load_current(self):
        """Load current configuration"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return json.load(f)
        return DEFAULT_PRESETS["baseline"]["settings"].copy()

    def load_presets(self):
        """Load saved presets"""
        if PRESETS_FILE.exists():
            with open(PRESETS_FILE) as f:
                return json.load(f)
        return DEFAULT_PRESETS.copy()

    def save_current(self):
        """Save current configuration"""
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.current, f, indent=2)

    def save_preset(self, name, description, reduction):
        """Save current config as a preset"""
        self.presets[name] = {
            "name": name,
            "description": description,
            "reduction": reduction,
            "settings": self.current.copy()
        }
        with open(PRESETS_FILE, 'w') as f:
            json.dump(self.presets, f, indent=2)

    def apply_preset(self, preset_key):
        """Apply a preset configuration"""
        if preset_key in self.presets:
            self.current = self.presets[preset_key]["settings"].copy()
            return True
        return False

    def calculate_reduction(self):
        """Calculate estimated reduction % compared to baseline"""
        baseline = DEFAULT_PRESETS["baseline"]["settings"]

        # Calculate activity reduction factors
        active_hours_baseline = baseline["active_end"] - baseline["active_start"]
        active_hours_current = self.current["active_end"] - self.current["active_start"]
        hours_factor = active_hours_current / active_hours_baseline

        # Sleep factor (inverse - longer sleep = less activity)
        sleep_avg_baseline = (baseline["min_sleep"] + baseline["default_sleep"] + baseline["max_sleep"]) / 3
        sleep_avg_current = (self.current["min_sleep"] + self.current["default_sleep"] + self.current["max_sleep"]) / 3
        sleep_factor = sleep_avg_baseline / sleep_avg_current

        # Reflection factor (cooldown and weight)
        reflection_cooldown_factor = baseline["reflection_cooldown"] / self.current["reflection_cooldown"]
        reflection_weight_factor = self.current["reflection_weight"] / baseline["reflection_weight"]
        reflection_factor = reflection_cooldown_factor * reflection_weight_factor

        # Chaos factor
        chaos_factor = 1.0
        if baseline["chaos_enabled"] and not self.current["chaos_enabled"]:
            chaos_factor = 0.9  # Chaos adds ~10% overhead
        elif self.current["chaos_enabled"]:
            chaos_factor = self.current["chaos_probability"] / baseline["chaos_probability"]

        # Thinking levels factor (estimate token usage)
        thinking_cost = {"think": 1.0, "think hard": 1.5, "think harder": 2.0}
        baseline_thinking_avg = sum(thinking_cost.get(v, 1.0) for v in baseline["thinking_levels"].values()) / 6
        current_thinking_avg = sum(thinking_cost.get(v, 1.0) for v in self.current["thinking_levels"].values()) / 6
        thinking_factor = current_thinking_avg / baseline_thinking_avg

        # Combined activity factor
        activity_factor = hours_factor * sleep_factor * reflection_factor * chaos_factor * thinking_factor

        # Convert to reduction percentage
        reduction = int((1 - activity_factor) * 100)
        return max(0, min(100, reduction))  # Clamp between 0-100


class ThrottleTUI:
    """Text-based UI for throttle control"""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.config = ThrottleConfig()
        self.menu_index = 0
        self.edit_mode = False
        self.edit_field = None
        self.edit_value = ""

        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)

        curses.curs_set(0)  # Hide cursor
        self.stdscr.keypad(True)

    def draw_header(self):
        """Draw header"""
        h, w = self.stdscr.getmaxyx()
        reduction = self.config.calculate_reduction()

        title = "CLAUDE DAEMON THROTTLE CONTROL"
        subtitle = f"Current Reduction: {reduction}% | Press 'q' to quit, 's' to save, 'p' for presets"

        self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(0, (w - len(title)) // 2, title)
        self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(1, (w - len(subtitle)) // 2, subtitle)
        self.stdscr.attroff(curses.color_pair(1))

        self.stdscr.addstr(2, 0, "─" * w)

    def draw_menu(self):
        """Draw main configuration menu"""
        h, w = self.stdscr.getmaxyx()
        y = 4

        menu_items = [
            ("Sleep Intervals", "sleep"),
            ("Active Hours", "hours"),
            ("Action Weights", "weights"),
            ("Reflection Settings", "reflection"),
            ("Chaos Settings", "chaos"),
            ("Thinking Levels", "thinking"),
            ("Calendar View", "calendar"),
            ("Apply Preset", "preset"),
            ("Save Custom Preset", "save"),
            ("Apply & Restart Daemon", "apply"),
        ]

        for idx, (label, key) in enumerate(menu_items):
            if idx == self.menu_index:
                self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                self.stdscr.addstr(y + idx, 2, f"> {label}")
                self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                self.stdscr.addstr(y + idx, 4, label)

        # Show current values in right column
        x_col2 = w // 2
        self.draw_current_values(y, x_col2)

    def draw_current_values(self, y, x):
        """Draw current configuration values"""
        c = self.config.current

        self.stdscr.attron(curses.color_pair(3))
        self.stdscr.addstr(y, x, "Current Settings:")
        self.stdscr.attroff(curses.color_pair(3))

        y += 1
        self.stdscr.addstr(y, x, f"Sleep: {c['min_sleep']//60}/{c['default_sleep']//60}/{c['max_sleep']//60} min")
        y += 1
        self.stdscr.addstr(y, x, f"Active: {c['active_start']:02d}:00 - {c['active_end']:02d}:00 EDT")
        y += 1
        self.stdscr.addstr(y, x, f"Weights: T{c['task_weight']:.1f} R{c['reflection_weight']:.1f} C{c['conversation_weight']:.1f}")
        y += 1
        self.stdscr.addstr(y, x, f"Reflection CD: {c['reflection_cooldown']} min")
        y += 1
        chaos_status = f"ON ({c['chaos_probability']:.0%})" if c['chaos_enabled'] else "OFF"
        self.stdscr.addstr(y, x, f"Chaos: {chaos_status}")
        y += 1

        # Thinking levels summary
        thinking_summary = f"{sum(1 for v in c['thinking_levels'].values() if v == 'think')}/6 std"
        self.stdscr.addstr(y, x, f"Thinking: {thinking_summary}")

    def draw_sleep_editor(self):
        """Draw sleep interval editor"""
        h, w = self.stdscr.getmaxyx()
        y = 4

        self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Sleep Intervals (minutes)")
        self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
        y += 2

        fields = [
            ("Morning (MIN_SLEEP)", "min_sleep", 60),
            ("Afternoon (DEFAULT_SLEEP)", "default_sleep", 60),
            ("Evening (MAX_SLEEP)", "max_sleep", 60),
            ("Night (NIGHT_SLEEP)", "night_sleep", 3600),
        ]

        for idx, (label, key, divisor) in enumerate(fields):
            value = self.config.current[key] // divisor

            if self.edit_mode and self.edit_field == key:
                self.stdscr.attron(curses.color_pair(2) | curses.A_REVERSE)
                display = f"{label}: {self.edit_value}_"
                self.stdscr.addstr(y + idx * 2, 4, display)
                self.stdscr.attroff(curses.color_pair(2) | curses.A_REVERSE)
            else:
                baseline = DEFAULT_PRESETS["baseline"]["settings"][key] // divisor
                diff = value - baseline
                diff_str = f" ({diff:+d})" if diff != 0 else ""

                marker = "> " if not self.edit_mode and idx == self.menu_index else "  "
                self.stdscr.addstr(y + idx * 2, 4, f"{marker}{label}: {value} min{diff_str}")

                # Visual bar
                bar_width = 40
                bar_fill = min(int((value / (baseline * 2)) * bar_width), bar_width)
                bar = "█" * bar_fill + "░" * (bar_width - bar_fill)
                self.stdscr.addstr(y + idx * 2 + 1, 6, bar)

        self.stdscr.addstr(h - 3, 2, "Use ↑↓ to navigate, ENTER to edit, +/- to adjust, ESC to cancel")

    def draw_hours_editor(self):
        """Draw active hours editor with calendar view"""
        h, w = self.stdscr.getmaxyx()
        y = 4

        self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Active Hours (EDT)")
        self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
        y += 2

        start = self.config.current["active_start"]
        end = self.config.current["active_end"]

        # Fields
        fields = [
            ("Start Hour", "active_start"),
            ("End Hour", "active_end"),
        ]

        for idx, (label, key) in enumerate(fields):
            value = self.config.current[key]

            if self.edit_mode and self.edit_field == key:
                self.stdscr.attron(curses.color_pair(2) | curses.A_REVERSE)
                display = f"{label}: {self.edit_value}_"
                self.stdscr.addstr(y + idx, 4, display)
                self.stdscr.attroff(curses.color_pair(2) | curses.A_REVERSE)
            else:
                marker = "> " if not self.edit_mode and idx == self.menu_index else "  "
                self.stdscr.addstr(y + idx, 4, f"{marker}{label}: {value:02d}:00")

        # Calendar visualization
        y += 4
        self.stdscr.attron(curses.color_pair(3))
        self.stdscr.addstr(y, 4, "24-Hour Activity Calendar:")
        self.stdscr.attroff(curses.color_pair(3))
        y += 1

        # Draw hour blocks
        for hour in range(24):
            x_pos = 4 + (hour % 12) * 5
            y_pos = y + (hour // 12)

            if start <= hour < end:
                # Active hour
                self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                self.stdscr.addstr(y_pos, x_pos, f"{hour:02d}")
                self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                # Inactive hour
                self.stdscr.attron(curses.color_pair(4))
                self.stdscr.addstr(y_pos, x_pos, f"{hour:02d}")
                self.stdscr.attroff(curses.color_pair(4))

        # Summary
        active_hours = end - start
        y += 3
        self.stdscr.addstr(y, 4, f"Active: {active_hours} hours/day ({active_hours/24*100:.0f}%)")

        self.stdscr.addstr(h - 3, 2, "Use ↑↓ to navigate, ENTER to edit, ESC to cancel")

    def draw_preset_selector(self):
        """Draw preset selector"""
        h, w = self.stdscr.getmaxyx()
        y = 4

        self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Select Preset Configuration")
        self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
        y += 2

        presets = list(self.config.presets.items())

        for idx, (key, preset) in enumerate(presets):
            if idx == self.menu_index:
                self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                self.stdscr.addstr(y + idx * 3, 4, f"> {preset['name']}")
                self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                self.stdscr.addstr(y + idx * 3, 6, preset['name'])

            self.stdscr.addstr(y + idx * 3 + 1, 8, preset['description'])

        self.stdscr.addstr(h - 3, 2, "Use ↑↓ to navigate, ENTER to apply, ESC to cancel")

    def run(self):
        """Main TUI loop"""
        mode = "menu"  # menu, sleep, hours, preset, etc.

        while True:
            self.stdscr.clear()
            self.draw_header()

            if mode == "menu":
                self.draw_menu()
            elif mode == "sleep":
                self.draw_sleep_editor()
            elif mode == "hours":
                self.draw_hours_editor()
            elif mode == "preset":
                self.draw_preset_selector()

            self.stdscr.refresh()

            # Handle input
            key = self.stdscr.getch()

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                self.save_custom_preset()
            elif key == ord('p') or key == ord('P'):
                mode = "preset"
                self.menu_index = 0
            elif key == curses.KEY_UP:
                if mode == "menu":
                    self.menu_index = max(0, self.menu_index - 1)
                elif mode in ("sleep", "hours", "preset"):
                    if not self.edit_mode:
                        self.menu_index = max(0, self.menu_index - 1)
            elif key == curses.KEY_DOWN:
                if mode == "menu":
                    self.menu_index = min(9, self.menu_index + 1)
                elif mode == "sleep":
                    if not self.edit_mode:
                        self.menu_index = min(3, self.menu_index + 1)
                elif mode == "hours":
                    if not self.edit_mode:
                        self.menu_index = min(1, self.menu_index + 1)
                elif mode == "preset":
                    if not self.edit_mode:
                        self.menu_index = min(len(self.config.presets) - 1, self.menu_index + 1)
            elif key == 10:  # ENTER
                if mode == "menu":
                    if self.menu_index == 0:
                        mode = "sleep"
                        self.menu_index = 0
                    elif self.menu_index == 1:
                        mode = "hours"
                        self.menu_index = 0
                    elif self.menu_index == 7:
                        mode = "preset"
                        self.menu_index = 0
                    elif self.menu_index == 9:
                        self.apply_and_restart()
                elif mode == "preset":
                    preset_key = list(self.config.presets.keys())[self.menu_index]
                    self.config.apply_preset(preset_key)
                    mode = "menu"
                    self.menu_index = 0
                elif mode == "sleep":
                    self.handle_sleep_edit()
                elif mode == "hours":
                    self.handle_hours_edit()
            elif key == 27:  # ESC
                if self.edit_mode:
                    self.edit_mode = False
                    self.edit_value = ""
                elif mode != "menu":
                    mode = "menu"
                    self.menu_index = 0

        self.config.save_current()

    def handle_sleep_edit(self):
        """Handle sleep interval editing"""
        fields = ["min_sleep", "default_sleep", "max_sleep", "night_sleep"]
        divisors = [60, 60, 60, 3600]

        field = fields[self.menu_index]
        divisor = divisors[self.menu_index]

        if not self.edit_mode:
            self.edit_mode = True
            self.edit_field = field
            self.edit_value = str(self.config.current[field] // divisor)
        else:
            try:
                new_value = int(self.edit_value) * divisor
                self.config.current[field] = new_value
                self.edit_mode = False
                self.edit_field = None
                self.edit_value = ""
            except ValueError:
                pass

    def handle_hours_edit(self):
        """Handle active hours editing"""
        fields = ["active_start", "active_end"]
        field = fields[self.menu_index]

        if not self.edit_mode:
            self.edit_mode = True
            self.edit_field = field
            self.edit_value = str(self.config.current[field])
        else:
            try:
                new_value = int(self.edit_value)
                if 0 <= new_value <= 23:
                    self.config.current[field] = new_value
                self.edit_mode = False
                self.edit_field = None
                self.edit_value = ""
            except ValueError:
                pass

    def save_custom_preset(self):
        """Save current config as custom preset"""
        # For now, auto-generate name
        reduction = self.config.calculate_reduction()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name = f"custom_{timestamp}"
        description = f"Custom configuration ({reduction}% reduction)"
        self.config.save_preset(name, description, reduction)

    def apply_and_restart(self):
        """Apply configuration and restart daemon"""
        self.config.save_current()
        # Generate daemon.sh with new settings
        self.generate_daemon_config()

        # Show confirmation
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        self.stdscr.addstr(h // 2, (w - 40) // 2, "Configuration applied! Restart daemon? (y/n)")
        self.stdscr.refresh()

        key = self.stdscr.getch()
        if key == ord('y') or key == ord('Y'):
            subprocess.run([str(DAEMON_ROOT / "daemon.sh"), "restart"])

    def generate_daemon_config(self):
        """Generate updated daemon.sh configuration"""
        # This would update daemon.sh with new values
        # For now, just save to config file
        pass


def main(stdscr):
    tui = ThrottleTUI(stdscr)
    tui.run()


if __name__ == "__main__":
    # Check if we should apply SAT preset
    if len(sys.argv) > 1 and sys.argv[1] == "--apply-sat":
        # Apply SAT configuration: 9-9 hours, +25% sleep
        config = ThrottleConfig()
        baseline = DEFAULT_PRESETS["baseline"]["settings"]

        # SAT settings
        config.current["min_sleep"] = int(baseline["min_sleep"] * 1.25)  # 750
        config.current["default_sleep"] = int(baseline["default_sleep"] * 1.25)  # 1125
        config.current["max_sleep"] = int(baseline["max_sleep"] * 1.25)  # 2250
        config.current["active_start"] = 9
        config.current["active_end"] = 21

        reduction = config.calculate_reduction()

        # Save as preset
        config.save_preset(
            f"SAT{reduction}",
            f"SAT: 9-9 hours, +25% sleep ({reduction}% reduction)",
            reduction
        )
        config.save_current()

        print(f"✓ SAT({reduction}%) configuration created and applied")
        print(f"  - Active hours: 9AM-9PM EDT (was 7AM-10PM)")
        print(f"  - Morning sleep: {config.current['min_sleep']//60} min (was {baseline['min_sleep']//60})")
        print(f"  - Afternoon sleep: {config.current['default_sleep']//60} min (was {baseline['default_sleep']//60})")
        print(f"  - Evening sleep: {config.current['max_sleep']//60} min (was {baseline['max_sleep']//60})")
        print(f"  - Estimated reduction: {reduction}%")
        print(f"\nRun without --apply-sat to open TUI for further adjustments")
    else:
        curses.wrapper(main)
