"""create shared conversation memory turns

Revision ID: 0001_shared_memory_turns
Revises:
Create Date: 2026-09-08

The agent's cross-turn memory: one row per completed turn, kept indefinitely.
Nothing in this schema expires a conversation — no TTL column, no expiry index,
nothing for a cleanup job to key on.

This schema keeps its own Alembic version table, independent of the Keycloak
role-mapping history in ``system_auth_nmd`` — one database, separate
schemas, separate histories, and neither can block the other.

``assistant-chat-nlp`` runs the same kind of memory against its own database,
not this one. Nothing here is shared with it: the two services keep separate
tables, separate histories and separate rows.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_shared_memory_turns"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "memory_shared"
TABLE = "conversation_memory_turns"
INDEX = "ix_conversation_memory_turns_conversation_id_id"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        TABLE,
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=128), nullable=False),
        sa.Column("payload_version", sa.SmallInteger(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversation_memory_turns"),
        schema=SCHEMA,
    )
    # Serves the only read there is: the last N turns of one conversation.
    # Postgres walks the index backwards for the DESC order, so ascending here
    # is enough and the query never sorts.
    op.create_index(
        INDEX,
        TABLE,
        ["conversation_id", "id"],
        unique=False,
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX, table_name=TABLE, schema=SCHEMA)
    op.drop_table(TABLE, schema=SCHEMA)
