from specrag.models import ResearchPlan, ResearchTask
from specrag.orchestration import (
    ResearchNodeHandlers,
    SQLiteCheckpointManager,
    build_single_research_workflow,
    new_research_state,
)


def _workflow(manager, marker):
    task = ResearchTask(task_id="T1", question="Q", objective="O")
    plan = ResearchPlan(plan_id="P1", tasks=(task,))
    return build_single_research_workflow(
        ResearchNodeHandlers(
            router=lambda state: {"route": "plan"},
            planner=lambda state: {"research_plan": plan, "research_tasks": plan.tasks},
            researcher=lambda state: {"iteration": 1},
            source_validator=lambda state: {},
            sufficiency_judge=lambda state: {"evidence_sufficient": True},
            replanner=lambda state: {},
            synthesizer=lambda state: {
                "final_answer": marker, "stop_reason": "completed",
            },
        ),
        checkpointer=manager.saver,
    )


def test_sqlite_checkpoint_survives_connection_restart(tmp_path) -> None:
    path = tmp_path / "checkpoint.sqlite3"
    first_manager = SQLiteCheckpointManager(path)
    first = _workflow(first_manager, "persisted answer")
    result = first.invoke(
        new_research_state(research_id="R1", user_question="Q"),
        thread_id="THREAD-persistent",
    )
    assert result["final_answer"] == "persisted answer"
    first_manager.close()

    second_manager = SQLiteCheckpointManager(path)
    try:
        second = _workflow(second_manager, "new answer should not replace checkpoint")
        snapshot = second.checkpoint(thread_id="THREAD-persistent")
        assert snapshot.values["final_answer"] == "persisted answer"
        assert snapshot.values["stop_reason"] == "completed"
    finally:
        second_manager.close()
