"""
Tool system for Club Harness.
"""

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.types import ToolCall, ToolResult


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


class Tool(ABC):
    """Abstract base class for tools."""

    name: str
    description: str
    parameters: List[ToolParameter] = []
    requires_confirmation: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against schema. Returns error message or None."""
        for param in self.parameters:
            if param.required and param.name not in arguments:
                return f"Missing required parameter: {param.name}"

            if param.name in arguments:
                value = arguments[param.name]
                # Basic type checking
                if param.type == "string" and not isinstance(value, str):
                    return f"Parameter {param.name} must be a string"
                if param.type == "number" and not isinstance(value, (int, float)):
                    return f"Parameter {param.name} must be a number"
                if param.type == "boolean" and not isinstance(value, bool):
                    return f"Parameter {param.name} must be a boolean"
                if param.enum and value not in param.enum:
                    return f"Parameter {param.name} must be one of: {param.enum}"

        return None


class CalculatorTool(Tool):
    """Calculator tool with safe math functions."""

    name = "calculator"
    description = "Perform arithmetic calculations with support for math functions (abs, min, max, round, sum, pow, sqrt)"
    parameters = [
        ToolParameter(
            name="expression",
            type="string",
            description="Mathematical expression to evaluate (e.g., '2 + 2', 'abs(-5)', 'sqrt(16)', 'max(1,2,3)')",
        ),
    ]

    # Safe math functions available for evaluation
    SAFE_FUNCTIONS = {
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'sum': sum,
        'pow': pow,
        'len': len,
        # Math module functions
        'sqrt': __import__('math').sqrt,
        'ceil': __import__('math').ceil,
        'floor': __import__('math').floor,
        'log': __import__('math').log,
        'log10': __import__('math').log10,
        'sin': __import__('math').sin,
        'cos': __import__('math').cos,
        'tan': __import__('math').tan,
        'pi': __import__('math').pi,
        'e': __import__('math').e,
    }

    def execute(self, expression: str, **kwargs) -> ToolResult:
        import time
        import re
        start = time.time()

        try:
            # Check for dangerous patterns
            dangerous_patterns = [
                r'__',           # Dunder methods
                r'import',       # Import statements
                r'exec',         # Exec function
                r'eval',         # Eval function
                r'open',         # File operations
                r'os\.',         # OS module
                r'sys\.',        # Sys module
                r'subprocess',   # Subprocess
                r'globals',      # Globals access
                r'locals',       # Locals access
                r'getattr',      # Attribute access
                r'setattr',      # Attribute setting
                r'delattr',      # Attribute deletion
                r'\[.*\]',       # List/dict subscript (allow only for basic usage)
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, expression, re.IGNORECASE):
                    return ToolResult(
                        success=False,
                        output=None,
                        error="Invalid characters in expression",
                        tool_name=self.name,
                    )

            # Replace ^ with ** for exponentiation
            expression = expression.replace("^", "**")

            # Evaluate with only safe functions available
            result = eval(expression, {"__builtins__": {}}, self.SAFE_FUNCTIONS)

            return ToolResult(
                success=True,
                output=result,
                execution_time=time.time() - start,
                tool_name=self.name,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start,
                tool_name=self.name,
            )


class ShellTool(Tool):
    """Execute shell commands (with confirmation required)."""

    name = "shell"
    description = "Execute a shell command"
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute",
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in seconds",
            required=False,
            default=30,
        ),
    ]
    requires_confirmation = True

    def execute(self, command: str, timeout: int = 30, **kwargs) -> ToolResult:
        import time
        start = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                execution_time=time.time() - start,
                tool_name=self.name,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout}s",
                execution_time=time.time() - start,
                tool_name=self.name,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start,
                tool_name=self.name,
            )


class WebSearchTool(Tool):
    """Simulated web search tool."""

    name = "web_search"
    description = "Search the web for information"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
        ),
        ToolParameter(
            name="num_results",
            type="number",
            description="Number of results to return",
            required=False,
            default=5,
        ),
    ]

    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        # This is a placeholder - in production, integrate with a real search API
        return ToolResult(
            success=True,
            output={
                "query": query,
                "results": [
                    {"title": f"Result {i+1} for '{query}'", "snippet": "..."}
                    for i in range(num_results)
                ],
                "note": "This is a simulated search result",
            },
            tool_name=self.name,
        )


class ToolRegistry:
    """
    Registry of available tools.

    Manages tool discovery, validation, and execution.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default tools."""
        self.register(CalculatorTool())
        self.register(ShellTool())
        self.register(WebSearchTool())

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI schemas for all tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        confirm_callback: Optional[Callable[[str, Dict], bool]] = None,
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments
            confirm_callback: Optional callback for confirmation
                             (tool_name, args) -> bool (True to proceed)
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
                tool_name=name,
            )

        # Validate arguments
        validation_error = tool.validate_arguments(arguments)
        if validation_error:
            return ToolResult(
                success=False,
                output=None,
                error=validation_error,
                tool_name=name,
            )

        # Confirmation if required
        if tool.requires_confirmation:
            if confirm_callback:
                if not confirm_callback(name, arguments):
                    return ToolResult(
                        success=False,
                        output=None,
                        error="Tool execution cancelled by user",
                        tool_name=name,
                    )
            else:
                # No callback provided, skip dangerous tools
                return ToolResult(
                    success=False,
                    output=None,
                    error="Tool requires confirmation but no callback provided",
                    tool_name=name,
                )

        # Execute
        return tool.execute(**arguments)

    def execute_tool_call(
        self,
        tool_call: ToolCall,
        confirm_callback: Optional[Callable[[str, Dict], bool]] = None,
    ) -> ToolResult:
        """Execute from a ToolCall object."""
        return self.execute(
            tool_call.name,
            tool_call.arguments,
            confirm_callback,
        )


# Global registry instance
tool_registry = ToolRegistry()
