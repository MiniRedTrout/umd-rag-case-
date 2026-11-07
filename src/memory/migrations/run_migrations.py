from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

CONFIG_PATH = Path(__file__).resolve().parents[1] / "alembic.ini"

#: Key for the PostgreSQL advisory lock every ``upgrade`` takes. Any 64-bit
#: integer works; this one is arbitrary and only has to stay stable, because
#: two processes agreeing on it is the entire mechanism. Taken as a
#: *transaction* lock inside the migration's own transaction (``env.py``), so
#: it is released by the commit or the rollback and cannot be leaked by a
#: process that dies mid-upgrade.
MIGRATION_LOCK_KEY = 0x5F0C_11E0


def memory_migration_config() -> Config:
    """Alembic configuration for the memory schema.

    Reading it opens no connection: ``env.py`` builds the URL from
    ``POSTGRES_*`` only when a command actually runs.

    Returns:
        Config — Alembic configuration.
    """
    return Config(str(CONFIG_PATH))


def memory_migration_heads() -> frozenset[str]:
    """Revisions the migration scripts in *this build* end at.

    What ``verify`` compares the database's stamp against. A set rather than a
    single revision so a branched history is a comparison rather than an
    exception thrown from inside a health check.

    Returns:
        frozenset[str] — Head revisions.
    """
    return frozenset(ScriptDirectory.from_config(memory_migration_config()).get_heads())


def run_memory_migrations(revision: str = "head") -> None:
    """Bring the conversation-memory schema up to ``revision``.

    Every process that starts runs this, and more than one of them may start at
    once. The serialisation is an advisory lock taken inside the migration's
    transaction — see ``MIGRATION_LOCK_KEY`` and ``env.py``: the second process
    waits, then finds there is nothing left to apply.

    Args:
        revision: str — Target revision; ``head`` by default.
    """
    command.upgrade(memory_migration_config(), revision)


if __name__ == "__main__":
    run_memory_migrations()
