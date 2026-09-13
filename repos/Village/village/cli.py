"""Command Line Interface for Village framework."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from village import Village, Villager
from village.utils.config import get_config, Config
from village.utils.logging import configure_logging, get_logger


console = Console()
logger = get_logger(__name__)


class MockLLMProvider:
    """Simple mock LLM provider for CLI demonstrations."""
    
    def __init__(self, name: str = "CLI-Mock"):
        self.name = name
        
    async def generate(self, prompt: str) -> str:
        """Generate a simple mock response."""
        return f"[{self.name}] Mock response to: {prompt[:50]}..."


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--log-level', default='INFO', help='Logging level')
def cli(config: Optional[str], debug: bool, log_level: str) -> None:
    """Village - LLM Interaction Framework CLI."""
    # Configure logging
    configure_logging(level=log_level)
    
    # Load configuration
    if config:
        global_config = Config(config)
    else:
        global_config = get_config()
        
    if debug:
        global_config.set('development.debug_mode', True)
        
    console.print(Panel("🏘️ Village CLI", subtitle="LLM Interaction Framework"))


@cli.command()
@click.option('--name', '-n', default='CLI Village', help='Village name')
@click.option('--villagers', '-v', multiple=True, help='Villager names to create')
@click.option('--task', '-t', help='Task to execute collaboratively')
def create(name: str, villagers: tuple, task: Optional[str]) -> None:
    """Create a new village with villagers."""
    asyncio.run(_create_village(name, villagers, task))


async def _create_village(name: str, villager_names: tuple, task: Optional[str]) -> None:
    """Async implementation of village creation."""
    village = Village(name)
    
    console.print(f"\n🏘️ Created village: [bold]{name}[/bold]")
    
    if not villager_names:
        villager_names = ('analyst', 'researcher', 'coordinator')
        
    # Create villagers
    for villager_name in villager_names:
        provider = MockLLMProvider(f"Mock-{villager_name}")
        villager = Villager(villager_name, provider, role=villager_name)
        village.add_villager(villager)
        
    _display_village_info(village)
    
    if task:
        console.print(f"\n📋 Executing task: [italic]{task}[/italic]")
        result = await village.collaborate(task)
        
        console.print("\n🤝 Collaboration Result:")
        console.print(Panel(result, title="Task Result"))


@cli.command()
@click.option('--villager', '-v', required=True, help='Villager name')
@click.option('--role', '-r', default='general', help='Villager role')
@click.option('--task', '-t', required=True, help='Task to execute')
def run_task(villager: str, role: str, task: str) -> None:
    """Run a task with a single villager."""
    asyncio.run(_run_single_task(villager, role, task))


async def _run_single_task(villager_name: str, role: str, task: str) -> None:
    """Async implementation of single task execution."""
    provider = MockLLMProvider(f"Mock-{villager_name}")
    villager = Villager(villager_name, provider, role=role)
    
    console.print(f"\n👤 Villager: [bold]{villager_name}[/bold] (Role: {role})")
    console.print(f"📋 Task: [italic]{task}[/italic]")
    
    result = await villager.process_task(task)
    
    console.print("\n✅ Result:")
    console.print(Panel(result, title=f"{villager_name} Response"))


@cli.command()
def info() -> None:
    """Display information about the Village framework."""
    _display_framework_info()


@cli.command()
@click.option('--format', '-f', default='table', type=click.Choice(['table', 'yaml']), help='Output format')
def config_show(format: str) -> None:
    """Show current configuration."""
    config = get_config()
    
    if format == 'yaml':
        import yaml
        console.print(yaml.dump(config.to_dict(), default_flow_style=False))
    else:
        _display_config_table(config)


@cli.command()
@click.argument('key')
@click.argument('value')
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    config = get_config()
    
    # Try to parse value as appropriate type
    if value.lower() in ('true', 'false'):
        parsed_value = value.lower() == 'true'
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value
                
    config.set(key, parsed_value)
    console.print(f"✅ Set {key} = {parsed_value}")


def _display_village_info(village: Village) -> None:
    """Display village information in a formatted table."""
    table = Table(title=f"Village: {village.name}")
    table.add_column("Villager", style="cyan")
    table.add_column("Role", style="magenta")
    table.add_column("Status", style="green")
    
    for villager_name in village.list_villagers():
        villager = village.get_villager(villager_name)
        status = "Ready" if villager.llm_provider else "No Provider"
        table.add_row(villager_name, villager.role, status)
        
    console.print(table)


def _display_framework_info() -> None:
    """Display framework information."""
    from village import __version__, __author__
    
    info_panel = f"""
[bold]Village Framework v{__version__}[/bold]
Created by: {__author__}

🏗️ Architecture: Modular LLM interaction framework
🤖 Villagers: Individual agents with specialized roles
🏘️ Villages: Collaborative agent communities
🔌 Providers: Pluggable LLM service abstractions

[dim]Use 'village --help' for available commands[/dim]
    """
    
    console.print(Panel(info_panel.strip(), title="Framework Information"))


def _display_config_table(config: Config) -> None:
    """Display configuration in table format."""
    table = Table(title="Village Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="green")
    
    config_dict = config.to_dict()
    
    for section, section_data in config_dict.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                table.add_row(section, key, str(value))
        else:
            table.add_row(section, "", str(section_data))
            
    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error("CLI error", error=str(e))
        console.print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()