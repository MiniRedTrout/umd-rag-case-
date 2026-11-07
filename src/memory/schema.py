from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "REQUIRED_COLUMNS",
    "TABLE",
    "WRITTEN_COLUMNS",
    "Column",
    "columns_query",
    "columns_from_rows",
    "shape_mismatch",
]

TABLE = "conversation_memory_turns"

REQUIRED_COLUMNS: dict[str, bool] = {
    "id": False,
    "conversation_id": False,
    "payload": False,
    "created_at": False,
}

WRITTEN_COLUMNS: frozenset[str] = frozenset({"conversation_id", "payload"})


@dataclass(frozen=True)
class Column:
    nullable: bool
    has_default: bool


def columns_query() -> str:
    return (
        "SELECT column_name, is_nullable, column_default, is_identity "
        "FROM information_schema.columns "
        "WHERE table_schema = :schema AND table_name = :table"
    )


def columns_from_rows(rows) -> dict[str, Column]:
    return {
        row[0]: Column(
            nullable=row[1] == "YES",
            has_default=row[2] is not None or row[3] == "YES",
        )
        for row in rows
    }


def shape_mismatch(columns: dict[str, Column]) -> str:
    for name, nullable in REQUIRED_COLUMNS.items():
        column = columns.get(name)
        if column is None:
            return f"the {name} column is missing"
        if column.nullable != nullable:
            expected = "nullable" if nullable else "NOT NULL"
            return f"the {name} column is not {expected}"
    unfillable = sorted(
        name
        for name, column in columns.items()
        if name not in WRITTEN_COLUMNS
        and not column.nullable
        and not column.has_default
    )
    if unfillable:
        return (
            f"{', '.join(unfillable)} {'is' if len(unfillable) == 1 else 'are'} "
            f"mandatory, and this service never writes "
            f"{'it' if len(unfillable) == 1 else 'them'}"
        )
    return ""