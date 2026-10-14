"""Product CLI dispatch for identity, workspace, research, facts, and memory."""

from __future__ import annotations

import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import sys
from typing import Any

from specrag.access import AuthorizationError
from specrag.config import load_config
from specrag.memory import resolve_user_configuration
from specrag.models import ResearchBudget, ResearchRequest

from . import diagnostics
from .workspace_services import WorkspaceServices


SESSION_FILE = Path(".local/state/session-token")
_DIAGNOSTIC_COMMANDS = {"route", "plan", "research-plan", "baseline"}


def _password(prompt: str = "Password: ") -> str:
    return os.environ.get("SPECRAG_PASSWORD") or getpass(prompt)


def _save_token(token: str) -> None:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(token, encoding="utf-8")
    try:
        os.chmod(SESSION_FILE, 0o600)
    except OSError:
        pass


def _token() -> str:
    from_environment = os.environ.get("SPECRAG_SESSION_TOKEN", "").strip()
    if from_environment:
        return from_environment
    if not SESSION_FILE.is_file():
        raise AuthorizationError("no local session; run `specrag auth login`")
    return SESSION_FILE.read_text(encoding="utf-8").strip()


def _context(services: WorkspaceServices, workspace: str):
    return services.identity.context(_token(), workspace)


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specrag")
    commands = parser.add_subparsers(dest="command", required=True)

    auth = commands.add_parser("auth", help="manage local authentication")
    auth_commands = auth.add_subparsers(dest="auth_command", required=True)
    for name in ("bootstrap", "login"):
        command = auth_commands.add_parser(name)
        command.add_argument("--username", required=True)
    auth_commands.add_parser("logout")

    workspace = commands.add_parser("workspace", help="manage workspaces")
    workspace_commands = workspace.add_subparsers(dest="workspace_command", required=True)
    create = workspace_commands.add_parser("create")
    create.add_argument("--name", required=True)
    member = workspace_commands.add_parser("member")
    member_commands = member.add_subparsers(dest="member_command", required=True)
    add = member_commands.add_parser("add")
    _workspace(add)
    add.add_argument("--username", required=True)
    add.add_argument("--role", choices=("member", "reviewer", "admin"), default="member")
    remove = member_commands.add_parser("remove")
    _workspace(remove)
    remove.add_argument("--username", required=True)
    role = member_commands.add_parser("role")
    _workspace(role)
    role.add_argument("--username", required=True)
    role.add_argument("--role", choices=("member", "reviewer", "admin"), required=True)

    research = commands.add_parser("research", help="run and manage research")
    research_commands = research.add_subparsers(dest="research_command", required=True)
    run = research_commands.add_parser("run")
    _workspace(run)
    run.add_argument("--question", required=True)
    run.add_argument("--release", type=int)
    run.add_argument("--language")
    run.add_argument("--detail")
    run.add_argument("--elapsed-seconds", type=int, default=900)
    run.add_argument("--max-iterations", type=int, default=4)
    run.add_argument("--max-tool-calls", type=int, default=12)
    run.add_argument("--max-gap-tool-calls", type=int, default=6)
    run.add_argument("--max-replans", type=int, default=3)
    for name in ("resume", "inspect", "cancel"):
        command = research_commands.add_parser(name)
        _workspace(command)
        command.add_argument("run_id")

    fact = commands.add_parser("specfact", help="review specification facts")
    fact_commands = fact.add_subparsers(dest="specfact_command", required=True)
    review = fact_commands.add_parser("review")
    _workspace(review)
    review.add_argument("proposal_id")
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    review.add_argument("--reason")

    memory = commands.add_parser("memory", help="manage private User Memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    set_command = memory_commands.add_parser("set")
    _workspace(set_command)
    set_command.add_argument("--kind", required=True, choices=(
        "default_release", "answer_language", "detail_level",
        "followed_topic", "research_lead",
    ))
    set_command.add_argument("--key", required=True)
    set_command.add_argument("--value", required=True)
    list_command = memory_commands.add_parser("list")
    _workspace(list_command)
    delete = memory_commands.add_parser("delete")
    _workspace(delete)
    delete.add_argument("memory_id")
    return parser


def _workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)


def _dispatch_auth(args: argparse.Namespace, services: WorkspaceServices) -> int:
    if args.auth_command == "bootstrap":
        password = _password()
        user = services.identity.bootstrap(args.username, password)
        token, expires = services.identity.login(args.username, password)
        _save_token(token)
        _json({
            "user": user.model_dump(mode="json"),
            "session_expires_at": expires.isoformat(),
        })
    elif args.auth_command == "login":
        token, expires = services.identity.login(args.username, _password())
        _save_token(token)
        _json({"username": args.username, "session_expires_at": expires.isoformat()})
    else:
        services.identity.logout(_token())
        SESSION_FILE.unlink(missing_ok=True)
        _json({"status": "logged_out"})
    return 0


def _dispatch_workspace(args: argparse.Namespace, services: WorkspaceServices) -> int:
    if args.workspace_command == "create":
        _json(services.identity.create_workspace(_token(), args.name))
        return 0
    context = _context(services, args.workspace)
    if args.member_command == "add":
        result = services.identity.add_user(
            context, args.username, _password("New user password: "), args.role
        )
    elif args.member_command == "remove":
        services.identity.remove_user(context, args.username)
        result = {"status": "removed", "username": args.username}
    else:
        result = services.identity.set_role(context, args.username, args.role)
    _json(result)
    return 0


def _inspection(value) -> dict[str, object]:
    return {
        "run": value.run.model_dump(mode="json"),
        "events": [item.model_dump(mode="json") for item in value.events],
        "stage_latency_ms": value.stage_latency_ms,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "checkpoint_available": value.checkpoint is not None,
    }


def _dispatch_research(args: argparse.Namespace, services: WorkspaceServices) -> int:
    context = _context(services, args.workspace)
    if args.research_command == "run":
        preferences = services.memory.list(context)
        config_release = load_config().retrieval.target_release or 17
        resolved = resolve_user_configuration(
            preferences,
            config_release=config_release,
            explicit_release=args.release,
            explicit_language=args.language,
            explicit_detail=args.detail,
        )
        request = ResearchRequest(
            question=args.question,
            release=resolved.release,
            answer_language=resolved.answer_language,
            detail_level=resolved.detail_level,
            elapsed_time_seconds=args.elapsed_seconds,
            budget=ResearchBudget(
                max_iterations=args.max_iterations,
                max_tool_calls=args.max_tool_calls,
                max_gap_tool_calls=args.max_gap_tool_calls,
                max_replans=args.max_replans,
            ),
        )
        result = services.harness.start(context, request)
    elif args.research_command == "resume":
        result = services.harness.resume(context, args.run_id)
    elif args.research_command == "cancel":
        _json(services.harness.cancel(context, args.run_id))
        return 0
    else:
        result = services.harness.inspect(context, args.run_id)
    _json(_inspection(result))
    return 0


def _dispatch(args: argparse.Namespace, services: WorkspaceServices) -> int:
    if args.command == "auth":
        return _dispatch_auth(args, services)
    if args.command == "workspace":
        return _dispatch_workspace(args, services)
    if args.command == "research":
        return _dispatch_research(args, services)
    context = _context(services, args.workspace)
    if args.command == "specfact":
        _json(services.specfacts.review(
            context, args.proposal_id, approve=args.approve, reason=args.reason
        ))
    elif args.memory_command == "set":
        _json(services.memory.set(
            context, kind=args.kind, key=args.key, value=args.value
        ))
    elif args.memory_command == "list":
        _json([item.model_dump(mode="json") for item in services.memory.list(context)])
    else:
        _json(services.memory.delete(context, args.memory_id))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in _DIAGNOSTIC_COMMANDS:
        return diagnostics.main(arguments)
    services: WorkspaceServices | None = None
    try:
        args = _parser().parse_args(arguments)
        services = WorkspaceServices.create()
        return _dispatch(args, services)
    except (AuthorizationError, KeyError, OSError, ValueError) as exc:
        print(f"specrag: {exc}", file=sys.stderr)
        return 2
    finally:
        if services is not None:
            services.close()


__all__ = ["main"]
