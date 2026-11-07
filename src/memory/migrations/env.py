import asyncio
import sys

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from py_gpn_postgres.common.env_wrapper import get_env_var, get_optional_env_var
from src.memory.config import SCHEMA, load_memory_database_config
from src.memory.entities import Base
from src.memory.migrations.run_migrations import MIGRATION_LOCK_KEY

config = context.config
target_metadata = Base.metadata

if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option(
        "sqlalchemy.url",
        # Escaped: a password may legitimately hold a '%', which configparser
        # would otherwise read as interpolation and mangle.
        load_memory_database_config().dsn.replace("%", "%%"),
    )


def include_name(name, type_, parent_names) -> bool:
    """Restrict autogenerate to the memory schema."""
    if type_ == "schema":
        return name in {SCHEMA}
    return True


def run_migrations_offline() -> None:
    """Emit SQL for the migration without connecting."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        version_table_schema=SCHEMA,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        # The schema has to exist before Alembic can place its version table
        # in it, exactly as the online path does — a revision cannot create
        # the schema its own version table lives in.
        context.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run the migrations on an open connection, serialised across replicas."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        # First statement in the transaction, before the schema is touched and
        # before Alembic reads the version table. Every process that starts
        # runs ``upgrade head``, and with more than one replica they run it at
        # the same time: without this both read "nothing applied" and both
        # apply it. The waiter proceeds only once the winner has committed, and
        # then reads a version table that says there is nothing left to do.
        #
        # The *transaction* variant rather than ``pg_advisory_lock`` and a
        # matching unlock: this one is released by the commit or the rollback,
        # so a process killed mid-upgrade cannot leave the lock held and every
        # later deployment blocked behind a session that no longer exists.
        context.execute(f"SELECT pg_advisory_xact_lock({MIGRATION_LOCK_KEY})")
        # The schema itself is created here rather than in a revision: a
        # revision cannot create the schema its own version table lives in.
        context.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
        context.run_migrations()


async def run_async_migrations() -> None:
    """Open a throwaway engine and run the migrations through it."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run the migrations against a live database."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if "true" == get_optional_env_var("POSTGRES_USE_KRB"):
        from krbcontext import krbContext

        with krbContext(
            using_keytab=True,
            principal=get_env_var("POSTGRES_KRB_PRINCIPAL"),
            keytab_file=get_env_var("POSTGRES_KRB_KEYTAB"),
        ):
            asyncio.run(run_async_migrations())
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
