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
from specrag.config import REPO_ROOT, load_config
from specrag.infrastructure import (
    StructuredMemoryConsolidator,
    StructuredMemoryExtractor,
    ThreeGppArchiveProvider,
    ThreeGppSourceMaterializer,
)
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.memory import resolve_user_configuration
from specrag.models import (
    MemoryExtractionSource,
    ResearchBudget,
    ResearchRequest,
)
from specrag.retrieval.corpus import discover_all_source_documents

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
    run.add_argument(
        "--elapsed-seconds",
        type=int,
        default=None,
        help=(
            "optional whole-Run wall-clock limit; omitted uses completion-first mode"
        ),
    )
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

    memory = commands.add_parser("memory", help="manage Core and Topic Memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    set_command = memory_commands.add_parser("set")
    _workspace(set_command)
    set_command.add_argument("--scope", choices=("user", "project"), default="user")
    set_command.add_argument("--kind", required=True, choices=(
        "default_release", "answer_language", "detail_level",
        "preference", "research_habit", "followed_topic", "research_lead",
        "project_decision", "evaluation_convention", "experiment_lesson",
    ))
    set_command.add_argument("--key", required=True)
    set_command.add_argument("--value", required=True)
    list_command = memory_commands.add_parser("list")
    _workspace(list_command)
    list_command.add_argument("--scope", choices=("user", "project"), default="user")
    delete = memory_commands.add_parser("delete")
    _workspace(delete)
    delete.add_argument("memory_id")

    topic = memory_commands.add_parser("topic")
    topic_commands = topic.add_subparsers(dest="topic_command", required=True)
    topic_create = topic_commands.add_parser("create")
    _workspace(topic_create)
    topic_create.add_argument("--slug", required=True)
    topic_create.add_argument("--description", required=True)
    topic_create.add_argument("--pinned", action="store_true")
    topic_list = topic_commands.add_parser("list")
    _workspace(topic_list)
    topic_add = topic_commands.add_parser("add")
    _workspace(topic_add)
    topic_add.add_argument("topic_id")
    topic_add.add_argument("--kind", required=True, choices=(
        "project_decision", "evaluation_convention", "experiment_lesson",
    ))
    topic_add.add_argument("--key", required=True)
    topic_add.add_argument("--content", required=True)
    for name, pinned in (("pin", True), ("unpin", False)):
        topic_pin = topic_commands.add_parser(name)
        _workspace(topic_pin)
        topic_pin.add_argument("topic_id")
        topic_pin.set_defaults(topic_pinned=pinned)

    proposal = memory_commands.add_parser("proposal")
    proposal_commands = proposal.add_subparsers(dest="proposal_command", required=True)
    proposal_list = proposal_commands.add_parser("list")
    _workspace(proposal_list)
    proposal_review = proposal_commands.add_parser("review")
    _workspace(proposal_review)
    proposal_review.add_argument("proposal_id")
    proposal_decision = proposal_review.add_mutually_exclusive_group(required=True)
    proposal_decision.add_argument("--approve", action="store_true")
    proposal_decision.add_argument("--reject", action="store_true")
    proposal_review.add_argument("--reason")

    recall = memory_commands.add_parser("recall")
    _workspace(recall)
    recall.add_argument("--query", required=True)
    extract = memory_commands.add_parser("extract")
    _workspace(extract)
    extract.add_argument("--source-ref", required=True)
    extract.add_argument("--user-text", required=True)
    extract.add_argument("--assistant-text", default="")

    maintenance = memory_commands.add_parser("maintenance")
    maintenance_commands = maintenance.add_subparsers(
        dest="maintenance_command", required=True
    )
    maintenance_status = maintenance_commands.add_parser("status")
    _workspace(maintenance_status)
    maintenance_status.add_argument(
        "--scope", choices=("user", "workspace"), required=True
    )
    maintenance_consolidate = maintenance_commands.add_parser("consolidate")
    _workspace(maintenance_consolidate)
    maintenance_consolidate.add_argument(
        "--scope", choices=("user", "workspace"), required=True
    )
    maintenance_prune = maintenance_commands.add_parser("prune")
    _workspace(maintenance_prune)
    maintenance_restore = maintenance_commands.add_parser("restore")
    _workspace(maintenance_restore)
    maintenance_restore.add_argument("memory_ids", nargs="+")

    corpus = commands.add_parser(
        "corpus", help="inspect or explicitly synchronize specification versions"
    )
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_status = corpus_commands.add_parser("status")
    corpus_status.add_argument("--release", type=int)
    corpus_sync = corpus_commands.add_parser("sync")
    corpus_sync.add_argument("--release", type=int)
    corpus_sync.add_argument(
        "--apply",
        action="store_true",
        help="build and activate candidates; without this flag the command is a dry run",
    )
    corpus_sync.add_argument(
        "--official",
        action="store_true",
        help="discover current versions from the official 3GPP archive",
    )
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
            configuration_memory_ids=resolved.memory_ids,
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


def _memory_agent() -> LLMApiStructuredAgent:
    return LLMApiStructuredAgent(
        role="summary", reasoning_effort="high", config=load_config()
    )


def _dispatch_memory(args: argparse.Namespace, services: WorkspaceServices) -> int:
    context = _context(services, args.workspace)
    if args.memory_command == "set":
        if args.scope == "project":
            result = services.memory.set_project_core(
                context, kind=args.kind, key=args.key, content=args.value
            )
        else:
            result = services.memory.set(
                context, kind=args.kind, key=args.key, value=args.value
            )
        _json(result)
    elif args.memory_command == "list":
        records = (
            services.memory.list_project_core(context)
            if args.scope == "project"
            else services.memory.list(context)
        )
        _json([item.model_dump(mode="json") for item in records])
    elif args.memory_command == "delete":
        _json(services.memory.delete(context, args.memory_id))
    elif args.memory_command == "topic":
        if args.topic_command == "create":
            result = services.memory.create_topic(
                context,
                slug=args.slug,
                description=args.description,
                pinned=args.pinned,
            )
        elif args.topic_command == "list":
            _json(
                [
                    {
                        **topic.model_dump(mode="json"),
                        "entries": [
                            item.model_dump(mode="json")
                            for item in services.memory.list_topic_entries(
                                context, topic.topic_id
                            )
                        ],
                    }
                    for topic in services.memory.list_topics(context)
                ]
            )
            return 0
        elif args.topic_command == "add":
            result = services.memory.add_topic_entry(
                context,
                args.topic_id,
                kind=args.kind,
                key=args.key,
                content=args.content,
            )
        else:
            result = services.memory.set_topic_pinned(
                context, args.topic_id, pinned=args.topic_pinned
            )
        _json(result)
    elif args.memory_command == "proposal":
        if args.proposal_command == "list":
            _json(
                [
                    item.model_dump(mode="json")
                    for item in services.memory_proposals.list(context)
                ]
            )
        else:
            _json(
                services.memory_proposals.review(
                    context,
                    args.proposal_id,
                    approve=args.approve,
                    reason=args.reason,
                )
            )
    elif args.memory_command == "recall":
        result = services.memory_recall.recall(context, args.query)
        _json(
            {
                "entries": [item.model_dump(mode="json") for item in result.entries],
                "pinned_topic_ids": result.pinned_topic_ids,
                "recalled_topic_ids": result.recalled_topic_ids,
                "omitted_entry_count": result.omitted_entry_count,
                "characters_used": result.characters_used,
            }
        )
    elif args.memory_command == "extract":
        extractor = StructuredMemoryExtractor(_memory_agent())
        records = services.memory_extraction.propose_from_source(
            context,
            MemoryExtractionSource(
                source_ref=args.source_ref,
                user_text=args.user_text,
                assistant_text=args.assistant_text,
            ),
            extractor,
        )
        _json([item.model_dump(mode="json") for item in records])
    elif args.maintenance_command == "status":
        _json(
            services.memory_maintenance.eligibility(
                context, scope=args.scope
            ).__dict__
        )
    elif args.maintenance_command == "consolidate":
        consolidator = StructuredMemoryConsolidator(_memory_agent())
        records = services.memory_maintenance.propose_consolidation(
            context, scope=args.scope, consolidator=consolidator
        )
        _json([item.model_dump(mode="json") for item in records])
    elif args.maintenance_command == "prune":
        records = services.memory_maintenance.propose_pruning(context)
        _json([item.model_dump(mode="json") for item in records])
    else:
        _json(
            services.memory_maintenance.propose_restore(
                context, *args.memory_ids
            )
        )
    return 0


def _dispatch(args: argparse.Namespace, services: WorkspaceServices) -> int:
    if args.command == "auth":
        return _dispatch_auth(args, services)
    if args.command == "workspace":
        return _dispatch_workspace(args, services)
    if args.command == "research":
        return _dispatch_research(args, services)
    if args.command == "memory":
        return _dispatch_memory(args, services)
    if args.command == "corpus":
        if args.corpus_command == "status":
            _json({
                "versions": [
                    item.model_dump(mode="json")
                    for item in services.corpus_store.list_versions(args.release)
                ],
                "active": [
                    item.model_dump(mode="json")
                    for release in (
                        (args.release,) if args.release is not None else
                        tuple(sorted({
                            item.release
                            for item in services.corpus_store.list_versions()
                        }))
                    )
                    for item in services.corpus_store.active_versions(release)
                ],
                "automatic_scheduler": "disabled",
                "physical_retirement": "disabled",
            })
            return 0
        app_config = load_config()
        cfg = app_config.retrieval
        supported = (
            {args.release}
            if args.release is not None
            else set(app_config.corpus_sync.supported_releases)
        )
        if args.official:
            provider = ThreeGppArchiveProvider(
                source_root_relative=cfg.specification_root,
            )
            materializer = ThreeGppSourceMaterializer(repository_root=REPO_ROOT)
            results = services.corpus_sync.synchronize_from_provider(
                provider,
                materializer,
                releases=tuple(sorted(supported)),
                apply=args.apply,
            )
        else:
            documents = tuple(
                document
                for document in discover_all_source_documents(cfg)
                if document.release in supported
            )
            if not documents:
                raise ValueError(
                    "no local specification documents match the requested Releases"
                )
            results = services.corpus_sync.synchronize(
                documents,
                apply=args.apply,
            )
        _json({
            "mode": "apply" if args.apply else "dry_run",
            "source": "official_3gpp_archive" if args.official else "local_markdown",
            "automatic_scheduler": "disabled",
            "old_versions_deleted": False,
            "items": [item.model_dump(mode="json") for item in results],
        })
        return 0
    context = _context(services, args.workspace)
    if args.command == "specfact":
        _json(services.specfacts.review(
            context, args.proposal_id, approve=args.approve, reason=args.reason
        ))
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
