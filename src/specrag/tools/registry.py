"""Provider-neutral registry for bounded agent-callable tools."""

from __future__ import annotations

from .models import (
    RetrievalTool,
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    ToolCapability,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RetrievalTool] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def capabilities(self) -> tuple[ToolCapability, ...]:
        return tuple(self._tools[name].capability for name in self.names)

    def schema_bundle(self) -> dict[str, object]:
        """Return framework-neutral schemas and capability metadata."""

        return {
            "request": RetrievalToolRequest.model_json_schema(),
            "result": RetrievalToolResult.model_json_schema(),
            "capabilities": [
                capability.model_dump(mode="json")
                for capability in self.capabilities
            ],
        }

    def register(self, tool: RetrievalTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool is already registered: {tool.name}")
        if tool.capability.name != tool.name:
            raise ValueError(
                "tool capability name does not match registered tool: "
                f"{tool.capability.name} != {tool.name}"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> RetrievalTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def invoke(
        self,
        name: RetrievalToolName,
        request: RetrievalToolRequest,
    ) -> RetrievalToolResult:
        result = self.get(name).invoke(request)
        if result.tool_name != name:
            raise RuntimeError(
                f"tool returned mismatched identity: {result.tool_name} != {name}"
            )
        return result


__all__ = ["ToolRegistry"]
