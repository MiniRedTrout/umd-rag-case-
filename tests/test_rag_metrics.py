from src.evaluation.rag_metrics import aggregate_metrics, evaluate_query
from src.rag.chunks import ChunkRecord


def _record(path: str, index: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_key=f"{path}:{index}",
        doc_id=path,
        path=path,
        chunk_index=index,
        text="text",
    )


def test_evaluate_query_uses_document_source_and_rank() -> None:
    result = evaluate_query(
        "q1",
        "course.docx",
        [_record("other.docx", 0), _record("course.docx", 1)],
        [1, 2, 5],
    )

    assert result.reciprocal_rank == 0.5
    assert result.hit_at == {1: False, 2: True, 5: True}
    assert result.precision_at == {1: 0.0, 2: 0.5, 5: 0.2}


def test_aggregate_metrics() -> None:
    first = evaluate_query("q1", "a.docx", [_record("a.docx", 0)], [1, 3])
    second = evaluate_query("q2", "a.docx", [_record("b.docx", 0)], [1, 3])

    assert aggregate_metrics([first, second]) == {
        "queries": 2,
        "mrr": 0.5,
        "recall_at_1": 0.5,
        "precision_at_1": 0.5,
        "recall_at_3": 0.5,
        "precision_at_3": 1 / 6,
    }

