"""Isolated Codex CLI adapter for schema-constrained agents."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .llm import LLMApiMode, OpenAIStructuredClient
from ..config import AppConfig, load_config


OutputT = TypeVar("OutputT", bound=BaseModel)


class CodexAgentError(RuntimeError):
    """Sanitized failure raised by the isolated Codex adapter."""


class CodexAgentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_runtime: str = "codex_cli"
    codex_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_ms: int = Field(ge=0)
    thread_id_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    sandbox: str = "read-only"
    ephemeral: bool = True


@dataclass(frozen=True)
class CodexAgentResponse(Generic[OutputT]):
    output: OutputT
    receipt: CodexAgentReceipt


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_codex_executable(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Codex executable not found: {candidate}")
        return candidate
    for name in ("codex.cmd", "codex.exe", "codex"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved).resolve()
    raise FileNotFoundError(
        "Codex CLI was not found on PATH; install it or pass an executable"
    )


def _codex_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unknown"


def _event_metadata(stdout: str) -> tuple[str, int | None, int | None]:
    thread_id = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started" and isinstance(
            event.get("thread_id"), str
        ):
            thread_id = event["thread_id"]
        usage = event.get("usage")
        if event.get("type") == "turn.completed" and isinstance(usage, dict):
            raw_input = usage.get("input_tokens")
            raw_output = usage.get("output_tokens")
            input_tokens = raw_input if isinstance(raw_input, int) else None
            output_tokens = raw_output if isinstance(raw_output, int) else None
    return thread_id, input_tokens, output_tokens


def _safe_cli_diagnostic(stderr: str, stdout: str) -> str:
    """Return a bounded, credential-redacted Codex CLI diagnostic."""

    source = "\n".join(
        value for value in (stderr.strip(), stdout.strip()) if value
    )
    if not source:
        return "no CLI diagnostic output"
    source = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", source)
    source = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)\S+",
        r"\1[REDACTED]",
        source,
    )
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    return " | ".join(lines[-8:])[-1600:]


class CodexStructuredAgent:
    """Run one ephemeral Codex turn in an isolated writable directory."""

    def __init__(
        self,
        *,
        work_root: Path,
        codex_executable: Path | None = None,
        model: str = "",
        reasoning_effort: str = "high",
        timeout_seconds: int = 600,
        sandbox: Literal["read-only", "workspace-write"] = "workspace-write",
        codex_home: Path | None = None,
    ) -> None:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self._work_root = work_root.resolve()
        self._executable = resolve_codex_executable(codex_executable)
        self._model = model.strip()
        self._reasoning_effort = reasoning_effort.strip()
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox
        self._codex_home = codex_home.resolve() if codex_home is not None else None

    def _prepare_work_root(self) -> None:
        """Fail early with a useful error when Windows ACLs block the agent."""

        try:
            self._work_root.mkdir(parents=True, exist_ok=True)
            probe = self._work_root / f".write-probe-{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            if probe.read_text(encoding="utf-8") != "ok":
                raise OSError("temporary workspace read-back failed")
            probe.unlink()
        except OSError as exc:
            raise CodexAgentError(
                "Codex agent work root is not writable; choose a fresh directory "
                f"inside the project (path={self._work_root}; "
                f"error={type(exc).__name__}: {exc})"
            ) from exc

    def _prepare_codex_home(self, child_env: dict[str, str]) -> None:
        """Verify that Codex's state directory is writable before launching."""

        configured = self._codex_home or (
            Path(child_env["CODEX_HOME"]).resolve()
            if child_env.get("CODEX_HOME")
            else Path.home() / ".codex"
        )
        try:
            configured.mkdir(parents=True, exist_ok=True)
            probe = configured / f".specrag-write-probe-{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise CodexAgentError(
                "Codex state home is not writable. Codex CLI needs to update "
                "state and tmp files even for --ephemeral runs. Run from a "
                "shell that can write the existing CODEX_HOME, or pass a "
                "writable authenticated codex_home "
                f"(path={configured}; error={type(exc).__name__}: {exc})"
            ) from exc
        if self._codex_home is not None:
            child_env["CODEX_HOME"] = str(configured)

    def parse(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: BaseModel | Mapping[str, Any],
        output_type: type[OutputT],
        allowed_read_paths: tuple[Path, ...] = (),
    ) -> CodexAgentResponse[OutputT]:
        for name, value in (
            ("purpose", purpose),
            ("prompt_version", prompt_version),
            ("instructions", instructions),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-blank")

        payload = (
            input_payload.model_dump(mode="json")
            if isinstance(input_payload, BaseModel)
            else dict(input_payload)
        )
        resolved_allowed_read_paths = tuple(
            path.resolve() for path in allowed_read_paths
        )
        request_text = _canonical_json({
            "purpose": purpose,
            "prompt_version": prompt_version,
            "instructions": instructions,
            "input": payload,
            "schema": output_type.model_json_schema(),
            "allowed_read_paths": [
                str(path) for path in resolved_allowed_read_paths
            ],
        })
        started = time.perf_counter()
        child_env = os.environ.copy()
        self._prepare_codex_home(child_env)
        self._prepare_work_root()
        # Use a fresh project-local directory for every call. This avoids old
        # Windows sandbox ACLs on .agent_tmp or a previous invocation from
        # poisoning a later run.
        workdir_path = self._work_root / f"specrag-codex-{uuid.uuid4().hex}"
        workdir_path.mkdir()
        with nullcontext(str(workdir_path)) as temporary_directory:
            workdir = Path(temporary_directory)
            runtime_temp = workdir / "runtime-temp"
            runtime_temp.mkdir()
            schema_path = workdir / "output_schema.json"
            # Keep the final response beside (not inside) the sandboxed cwd so
            # the parent can still read diagnostics even if a third-party
            # sandbox implementation unexpectedly changes the child ACL.
            output_path = self._work_root / f"{workdir.name}-final_response.json"
            (workdir / "AGENTS.md").write_text(
                instructions.rstrip() + "\n",
                encoding="utf-8",
            )
            (workdir / "task.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            schema_path.write_text(
                json.dumps(
                    output_type.model_json_schema(),
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            command = [
                str(self._executable),
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                self._sandbox,
                "--skip-git-repo-check",
                "--color",
                "never",
                "--json",
                "--cd",
                str(workdir),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if self._model:
                command.extend(("--model", self._model))
            if self._reasoning_effort:
                command.extend((
                    "--config",
                    f"model_reasoning_effort={self._reasoning_effort}",
                ))
            if resolved_allowed_read_paths:
                allowed_text = ", ".join(
                    str(path) for path in resolved_allowed_read_paths
                )
                command.append(
                    "Read task.json and follow AGENTS.md. You may inspect only "
                    f"these explicitly allowed read paths: {allowed_text}. "
                    "Use local search commands when requested. Return only JSON "
                    "that conforms to output_schema.json. Do not inspect any "
                    "other files and do not write or modify the allowed corpus."
                )
            else:
                command.append(
                    "Read task.json, follow AGENTS.md, and return only JSON that "
                    "conforms to output_schema.json. Do not inspect other files."
                )
            try:
                child_env["TEMP"] = str(runtime_temp)
                child_env["TMP"] = str(runtime_temp)
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._timeout_seconds,
                    env=child_env,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexAgentError(
                    f"Codex agent timed out after {self._timeout_seconds}s"
                ) from exc
            except OSError as exc:
                raise CodexAgentError(
                    f"Codex agent could not start ({type(exc).__name__})"
                ) from exc
            try:
                if completed.returncode != 0:
                    raise CodexAgentError(
                        "Codex agent failed "
                        f"(exit_code={completed.returncode}; "
                        f"diagnostic={_safe_cli_diagnostic(completed.stderr, completed.stdout)})"
                    )
                if not output_path.is_file():
                    raise CodexAgentError("Codex agent produced no final response")
                try:
                    raw_output = output_path.read_text(encoding="utf-8")
                    output = output_type.model_validate_json(raw_output)
                except ValueError as exc:
                    raise CodexAgentError(
                        "Codex agent returned invalid structured output"
                    ) from exc
            finally:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    pass
                # Only remove the unique directory created by this call. A
                # best-effort cleanup keeps prompts out of the project tree;
                # failure is harmless and will not poison the next call
                # because every invocation uses a new UUID path.
                if (
                    workdir.parent == self._work_root
                    and workdir.name.startswith("specrag-codex-")
                ):
                    try:
                        shutil.rmtree(workdir)
                    except OSError:
                        pass

        thread_id, input_tokens, output_tokens = _event_metadata(
            completed.stdout
        )
        response_text = _canonical_json(output)
        return CodexAgentResponse(
            output=output,
            receipt=CodexAgentReceipt(
                codex_version=_codex_version(self._executable),
                model=self._model or "codex_managed_default",
                reasoning_effort=(
                    self._reasoning_effort or "codex_managed_default"
                ),
                purpose=purpose,
                prompt_version=prompt_version,
                request_sha256=_sha256(request_text),
                response_sha256=_sha256(response_text),
                latency_ms=max(
                    0,
                    int((time.perf_counter() - started) * 1_000),
                ),
                thread_id_sha256=_sha256(thread_id) if thread_id else "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                sandbox=self._sandbox,
            ),
        )


class LLMApiStructuredAgent:
    """API-backed implementation with the same parse interface as Codex.

    This backend is deliberately limited to structured LLM calls.  It cannot
    inspect local files or execute tools, so the deterministic retrieval code
    remains in control of all corpus access and candidate selection.
    """

    def __init__(
        self,
        *,
        model: str = "",
        role: Literal["planning", "summary"] = "summary",
        reasoning_effort: str = "high",
        timeout_seconds: int = 600,
        api_mode: LLMApiMode = LLMApiMode.RESPONSES,
        config: AppConfig | None = None,
    ) -> None:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        settings = config or load_config()
        secrets = settings.secrets
        role_model = (
            secrets.openai_planning_model
            if role == "planning"
            else secrets.openai_summary_model
        ).strip()
        selected_model = (
            role_model
            or secrets.openai_model.strip()
            or model.strip()
        )
        if not selected_model:
            raise ValueError(
                "API backend requires OPENAI_MODEL, a role-specific model, "
                "or an explicit model"
            )
        self._model = selected_model
        self._reasoning_effort = reasoning_effort.strip() or "high"
        self._client = OpenAIStructuredClient(
            api_key=secrets.openai_api_key,
            base_url=secrets.openai_base_url,
            model=selected_model,
            api_mode=api_mode,
            reasoning_effort=self._reasoning_effort,
            timeout_seconds=float(timeout_seconds),
            max_retries=settings.llm.api_max_retries,
            fallback_models=(),
            retry_initial_delay_seconds=(
                settings.llm.api_retry_initial_delay_seconds
            ),
            retry_max_delay_seconds=settings.llm.api_retry_max_delay_seconds,
        )

    def parse(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: BaseModel | Mapping[str, Any],
        output_type: type[OutputT],
        allowed_read_paths: tuple[Path, ...] = (),
    ) -> CodexAgentResponse[OutputT]:
        if allowed_read_paths:
            raise CodexAgentError(
                "LLM API backend cannot inspect local files; disable the "
                "Agent evidence-locator when using --llm-backend api"
            )
        started = time.perf_counter()
        try:
            response = self._client.parse(
                purpose=purpose,
                prompt_version=prompt_version,
                instructions=instructions,
                input_payload=input_payload,
                output_type=output_type,
            )
        except Exception as exc:  # noqa: BLE001 - keep provider details bounded
            raise CodexAgentError(
                f"LLM API structured call failed ({type(exc).__name__}: {exc})"
            ) from None

        api_receipt = response.receipt
        return CodexAgentResponse(
            output=response.output,
            receipt=CodexAgentReceipt(
                agent_runtime="llm_api",
                codex_version="api",
                model=api_receipt.model,
                reasoning_effort=self._reasoning_effort,
                purpose=api_receipt.purpose,
                prompt_version=api_receipt.prompt_version,
                request_sha256=api_receipt.request_sha256,
                response_sha256=api_receipt.response_sha256,
                latency_ms=max(
                    api_receipt.latency_ms,
                    int((time.perf_counter() - started) * 1_000),
                ),
                input_tokens=api_receipt.input_tokens,
                output_tokens=api_receipt.output_tokens,
                sandbox="api",
                ephemeral=True,
            ),
        )


def build_structured_agent(
    *,
    backend: Literal["codex", "api"],
    work_root: Path,
    model: str,
    role: Literal["planning", "summary"],
    reasoning_effort: str,
    timeout_seconds: int,
    codex_executable: Path | None = None,
    codex_home: Path | None = None,
    sandbox: Literal["read-only", "workspace-write"] = "workspace-write",
) -> CodexStructuredAgent | LLMApiStructuredAgent:
    """Construct the selected backend without changing caller contracts."""

    if backend == "api":
        return LLMApiStructuredAgent(
            model=model,
            role=role,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
        )
    return CodexStructuredAgent(
        work_root=work_root,
        codex_executable=codex_executable,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        sandbox=sandbox,
        codex_home=codex_home,
    )


__all__ = [
    "CodexAgentError",
    "CodexAgentReceipt",
    "CodexAgentResponse",
    "CodexStructuredAgent",
    "LLMApiStructuredAgent",
    "build_structured_agent",
    "resolve_codex_executable",
]
