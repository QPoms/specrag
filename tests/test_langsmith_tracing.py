from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from specrag.infrastructure.langsmith_tracing import (
    LangSmithTracing,
    build_langsmith_tracing,
    wrap_openai_client,
)
from specrag.orchestration.application import ResearchApplication
from specrag.orchestration.parallel import ParallelResearchWorkflow
from specrag.orchestration.tracing import WorkflowTraceContext, workflow_config


def test_disabled_tracing_is_a_true_noop_and_hides_secret() -> None:
    loaded = False

    def loader():
        nonlocal loaded
        loaded = True
        raise AssertionError("disabled tracing must not load LangSmith")

    tracing = LangSmithTracing(
        enabled=False,
        project_name="specrag",
        _api_key="secret-key",
        _runtime_loader=loader,
    )
    with tracing.activate():
        pass

    assert loaded is False
    assert "secret-key" not in repr(tracing)
    client = object()
    assert wrap_openai_client(client, enabled=False) is client


def test_enabled_tracing_uses_injected_client_and_context_without_leaking() -> None:
    calls: dict[str, object] = {}

    def client_factory(**kwargs):
        calls["client_kwargs"] = kwargs
        return "client"

    @contextmanager
    def tracing_context(**kwargs):
        calls["context_kwargs"] = kwargs
        calls["entered"] = True
        yield
        calls["exited"] = True

    tracing = LangSmithTracing(
        enabled=True,
        project_name="comparison",
        _api_key="secret-key",
        _endpoint="https://langsmith.invalid",
        _runtime_loader=lambda: (client_factory, tracing_context),
    )
    with tracing.activate(tags=("phase:judge",), metadata={"question_id": "q1"}):
        assert calls["entered"] is True

    assert calls["client_kwargs"] == {
        "api_key": "secret-key",
        "api_url": "https://langsmith.invalid",
    }
    assert calls["context_kwargs"] == {
        "enabled": True,
        "project_name": "comparison",
        "client": "client",
        "tags": ["phase:judge"],
        "metadata": {"question_id": "q1"},
    }
    assert calls["exited"] is True
    assert "secret-key" not in repr(tracing)


def test_build_tracing_requires_both_flag_and_key() -> None:
    def config(*, enabled: bool, key: str):
        return SimpleNamespace(secrets=SimpleNamespace(
            langsmith_tracing=enabled,
            langsmith_api_key=key,
            langsmith_project="experiment",
            langsmith_endpoint="",
            langsmith_workspace_id="",
        ))

    assert build_langsmith_tracing(config(enabled=False, key="key")).enabled is False
    assert build_langsmith_tracing(config(enabled=True, key="")).enabled is False
    active = build_langsmith_tracing(config(enabled=True, key="key"))
    assert active.enabled is True
    assert active.project_name == "experiment"


def test_preflight_requires_trace_write_and_read_round_trip() -> None:
    calls: list[str] = []

    class Client:
        def create_run(self, name, inputs, run_type, **kwargs):
            assert name == "specrag.langsmith.preflight"
            assert inputs == {"probe": "trace-write-read"}
            assert run_type == "chain"
            calls.append("create")
            self.run_id = kwargs["id"]

        def flush(self, *, timeout):
            assert timeout == 10.0
            calls.append("flush")

        def read_run(self, run_id):
            assert run_id == self.run_id
            calls.append("read")
            return SimpleNamespace(id=run_id)

    client = Client()
    tracing = LangSmithTracing(
        enabled=True,
        project_name="comparison",
        _api_key="secret-key",
        _workspace_id="workspace-id",
        _runtime_loader=lambda: (
            lambda **kwargs: client,
            contextmanager(lambda: (yield))(),
        ),
    )

    result = tracing.preflight()

    assert result.status == "ready"
    assert calls == ["create", "flush", "read"]


def test_preflight_classifies_wrapped_http_403_without_leaking_details() -> None:
    def client_factory(**kwargs):
        class Client:
            def create_run(self, *args, **kwargs):
                raise RuntimeError("LangSmith request failed with HTTP 403")

        return Client()

    tracing = LangSmithTracing(
        enabled=True,
        project_name="comparison",
        _api_key="secret-key",
        _runtime_loader=lambda: (
            client_factory,
            contextmanager(lambda: (yield))(),
        ),
    )

    result = tracing.preflight()

    assert result.status == "unavailable"
    assert result.failure_kind == "authentication_or_configuration"
    assert result.retryable is False


def test_workflow_trace_context_reaches_langgraph_root_config() -> None:
    class Graph:
        def __init__(self) -> None:
            self.config = None

        def invoke(self, state, *, config):
            self.config = config
            return state

    graph = Graph()
    workflow = ParallelResearchWorkflow(graph=graph)
    trace = WorkflowTraceContext(
        run_id=uuid4(),
        run_name="workflow:gpt:test",
        tags=("phase:workflow", "model:gpt"),
        metadata={"experiment_id": "exp-1", "trial": 1},
    )
    state = {"schema_version": "research-state/v1.11"}
    workflow.invoke(state, thread_id="thread-1", trace_context=trace)

    assert graph.config == workflow_config("thread-1", trace)
    assert graph.config["configurable"] == {"thread_id": "thread-1"}
    assert graph.config["run_id"] == trace.run_id
    assert graph.config["metadata"]["experiment_id"] == "exp-1"


def test_research_application_activates_trace_around_workflow() -> None:
    events: list[str] = []

    class Tracing:
        @contextmanager
        def activate(self):
            events.append("trace-enter")
            yield
            events.append("trace-exit")

    class Workflow:
        def invoke(self, state, *, thread_id, trace_context):
            events.append("workflow")
            assert thread_id == "thread-1"
            assert trace_context is None
            return state

    app = ResearchApplication(
        workflow=Workflow(),
        release=17,
        corpus_scope=None,
        corpus_documents=(),
        retrieval_index_fingerprint="index",
        tracing=Tracing(),
    )
    result = app.run(
        research_id="research-1",
        question="Question",
        thread_id="thread-1",
    )

    assert result["research_id"] == "research-1"
    assert events == ["trace-enter", "workflow", "trace-exit"]
