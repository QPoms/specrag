"""Provider-neutral registry for bounded agent-callable tools."""

from __future__ import annotations

from specrag.models import NativeFunctionCall, NativeFunctionDefinition

from .models import (
    RetrievalTool,
    RetrievalToolArguments,
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    ToolCapability,
    parse_retrieval_tool_arguments,
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

    @property
    def definitions(self) -> tuple[NativeFunctionDefinition, ...]:
        """Expose strict provider-neutral native function definitions."""

        return tuple(
            NativeFunctionDefinition(
                name=tool.name,
                description=tool.capability.description,
                parameters=tool.argument_type.model_json_schema(),
            )
            for name in self.names
            if (tool := self._tools[name])
        )

    def schema_bundle(self) -> dict[str, object]:
        """Return framework-neutral schemas and capability metadata."""

        return {
            "tools": [
                definition.model_dump(mode="json")
                for definition in self.definitions
            ],
            "result": RetrievalToolResult.model_json_schema(),
            "capabilities": [
                capability.model_dump(mode="json")
                for capability in self.capabilities
            ],
        }

    def parse_call(self, call: NativeFunctionCall) -> RetrievalToolArguments:
        """Validate one native call with the registered tool's own schema."""

        self.get(call.name)
        try:
            return parse_retrieval_tool_arguments(call.name, call.arguments)
        except ValueError as exc:
            errors = getattr(exc, "errors", None)
            if not callable(errors):
                raise
            messages = "; ".join(
                error["msg"] for error in errors(include_url=False)
            )
            raise ValueError(
                f"invalid arguments for {call.name}: {messages}"
            ) from None

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
