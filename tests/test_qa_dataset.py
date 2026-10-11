from __future__ import annotations

from collections import Counter

from scripts.check_qa_dataset import validate_question_set


def test_r17_question_set_has_expected_shape() -> None:
    records = validate_question_set()

    assert len(records) == 50
    assert Counter(record["difficulty"] for record in records) == {
        "simple": 34,
        "complex": 16,
    }
    assert Counter(record["reasoning_scope"] for record in records) == {
        "single_clause": 30,
        "cross_clause": 4,
        "cross_specification": 16,
    }
    assert Counter(record["split"] for record in records) == {
        "development": 50,
    }
    assert all(record["source"]["dataset"] == "MM-TelcoBench" for record in records)
    assert len({
        (
            record["source"]["dataset"],
            record["source"]["record_kind"],
            record["source"]["record_id"],
        )
        for record in records
    }) == 50
