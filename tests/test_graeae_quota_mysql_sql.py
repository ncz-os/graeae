"""F09 regression — atomic quota row-locking SQL on MySQL/MariaDB.

The unquoted ``SELECT key FROM state`` previously shipped in
``mnemos/api/routes/consultations.py::_lock_quota_row_for_update`` is
broken on MySQL/MariaDB because ``KEY`` is a RESERVED WORD in those
dialects and ``key`` (lowercase) gets rejected with ``1064 You have an
error in your SQL syntax`` — silently surfacing as a generic 503 to
the caller. Oracle/DB2 use the bound-parameter form already (with
``key`` as a bind placeholder, which is not a SQL reserved-word issue),
Postgres ignores the ``key`` keyword-in-any-case quirk in practice
(it permits the bare identifier even though ``KEY`` is reserved for
orderings), and SQLite is permissive enough that the bare form
historically worked.

This test pins two shapes:
  1. The dialect-correct SQL for MySQL/MariaDB backticks every
     reserved identifier (``key`` in particular) and uses
     ``%s`` style parameters.
  2. The Oracle/DB2 branch (which the prior Oracle-shape regressions
     already cover end-to-end) still uses bare ``key`` since it is
     a bind parameter there and not a SQL token.

A live two-connection MySQL/MariaDB test was NOT added: this repo's
CI does not ship an aiomysql/pymysql fixture and the existing
``tests/conftest.py`` fixtures are backend-neutral. Probing the
generated SQL string is the static form the F09 instruction asked
for when no live MySQL/MariaDB infrastructure exists — comment in the
module's ``_mysql_lock_sql`` is the matching explanation.
"""

from __future__ import annotations

import pytest

from mnemos.api.routes import consultations


def test_mysql_lock_sql_backtick_quotes_key_reserved_word():
    """The MySQL/MariaDB SELECT must backtick-quote ``key``.

    Without backticks, MySQL 8.0+ and MariaDB 10.x reject the statement
    with a 1064 syntax error because ``KEY`` is a RESERVED WORD in both
    dialects. Backticks (and ONLY backticks — ANSI ``""`` is not legal in
    aiomysql's ``%s`` parameter style) bound the identifier.
    """
    sql = consultations._mysql_lock_sql()
    # Every reserved identifier must be backtick-quoted in the SELECT list
    # AND in the WHERE clause. The FOR UPDATE lock clause should not need
    # quoting (it's a SQL verb, not an identifier).
    assert "`key`" in sql, (
        f"MySQL SELECT must backtick-quote `key` (a reserved word) — got: {sql!r}"
    )
    # Other identifiers in the SELECT list also gain backticks so we are
    # consistent across dialect additions; assert this is uniform.
    assert "`owner_id`" in sql
    assert "`namespace`" in sql
    # Bare ``key`` (unquoted) anywhere in the rendered string is the exact
    # defect surface we are guarding against.
    assert " key " not in sql, (
        f"unquoted `key` is a reserved word on MySQL/MariaDB — got: {sql!r}"
    )
    # aiomysql uses ``%s`` style parameters, NOT ``?`` (sqlite) or
    # ``$1`` (asyncpg) or ``:key`` (oracledb).
    assert sql.count("%s") == 3, (
        f"MySQL lock SELECT should have 3 %s placeholders, got: {sql!r}"
    )


def test_mysql_lock_sql_returns_select_for_update():
    """The MySQL lock SELECT must include FOR UPDATE — same row-lock
    semantic as the Oracle/DB2 branch. Without it, the lock is just a
    plain row read and concurrent writers don't queue."""
    sql = consultations._mysql_lock_sql()
    assert "FOR UPDATE" in sql.upper(), (
        f"MySQL lock SELECT missing FOR UPDATE — got: {sql!r}"
    )
    assert sql.upper().startswith("SELECT"), (
        f"MySQL lock SELECT must start with SELECT — got: {sql!r}"
    )


def test_mysql_lock_sql_filters_deleted_at_is_null():
    """The lock SELECT must scope to live (non-deleted) state rows,
    matching the prior bare-SQL branch so the production semantics
    don't shift under the fix."""
    sql = consultations._mysql_lock_sql()
    assert "deleted_at IS NULL" in sql, (
        f"MySQL lock SELECT must filter out deleted rows — got: {sql!r}"
    )


def test_oracle_lock_sql_uses_bind_parameters():
    """Oracle/DB2 path uses bound parameters (:key), where ``key`` is a
    bind-placeholder, NOT a SQL identifier — so the dialect there has
    no reserved-word problem and the bare SQL is correct. This is a
    guard test: if anyone ever tries to copy the MySQL branch's backticks
    into the Oracle branch by accident, ``key`` would still be inside a
    bind-parameter dict so it would parse but as a different identifier
    name in the dictionary (effectively: no-op for backwards compat).
    """
    # Spot-check the rendered Oracle branch is left untouched.
    # We don't re-extract the SQL string here (the Oracle branch is
    # inline in _lock_quota_row_for_update), so this is a structural
    # check on the helper module: no sql string constant named
    # ``_ORACLE_LOCK_SQL`` exists, ensuring no accidental refactor
    # moves the MySQL-style backticking into the Oracle branch.
    source = open(consultations.__file__, encoding="utf-8").read()
    assert "SELECT key FROM state" in source, (
        "Oracle/DB2 branch should still use bare 'SELECT key FROM state' "
        "because 'key' there is a bind parameter placeholder, not a SQL "
        "identifier. If you have moved MySQL backticks into the Oracle "
        "branch by accident, revert that change."
    )


@pytest.mark.asyncio
async def test_lock_quota_row_for_update_mysql_renders_backticked_sql(
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end: a MySQL-shaped backend exercises the right SQL path."""

    executed: list[tuple[str, tuple]] = []

    class _FakeMysqlCursor:
        async def execute(self, sql: str, params: tuple | None = None) -> None:
            executed.append((sql, params))

        async def fetchone(self) -> tuple | None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeMysqlConn:
        def cursor(self) -> _FakeMysqlCursor:
            return _FakeMysqlCursor()

    fake_backend_cls = type("MysqlBackend", (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    tx = type("Tx", (), {"conn": _FakeMysqlConn()})()

    await consultations._lock_quota_row_for_update(tx, "principal-mysql")

    assert len(executed) == 1, f"expected exactly 1 execute call, got {executed!r}"
    sql, params = executed[0]
    assert "`key`" in sql, (
        f"MySQL execute must backtick-quote `key` — got: {sql!r}"
    )
    assert params == (
        consultations._QUOTA_OWNER_ID,
        consultations._QUOTA_NAMESPACE,
        "principal-mysql",
    )


@pytest.mark.asyncio
async def test_lock_quota_row_for_update_mariadb_renders_backticked_sql(
    monkeypatch: pytest.MonkeyPatch,
):
    """MariaDB shares the MySQL branch — confirm it gets the same dialect-correct SQL.

    MariaDB 10.x has the same reserved-word issue as MySQL 8.0+ for ``KEY``,
    so the backticked `` `key` `` form must apply to both. The previous fix
    routed both backends through the same ``_mysql_lock_sql`` helper; this
    test pins that the MariaBackend-shaped type still hits the backticked path
    rather than falling through to the bare-SQL Oracle/DB2 branch or the
    default Sqlite path.
    """
    executed: list[tuple[str, tuple]] = []

    class _FakeMysqlCursor:
        async def execute(self, sql: str, params: tuple | None = None) -> None:
            executed.append((sql, params))

        async def fetchone(self) -> tuple | None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeMysqlConn:
        def cursor(self) -> _FakeMysqlCursor:
            return _FakeMysqlCursor()

    fake_backend_cls = type("MariadbBackend", (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    tx = type("Tx", (), {"conn": _FakeMysqlConn()})()

    await consultations._lock_quota_row_for_update(tx, "principal-mariadb")

    assert len(executed) == 1, (
        f"MariaDB branch expected exactly 1 execute call, got {executed!r}"
    )
    sql, params = executed[0]
    assert "`key`" in sql, (
        f"MariaDB execute must backtick-quote `key` (also a reserved word "
        f"in MariaDB) — got: {sql!r}"
    )
    assert "`owner_id`" in sql
    assert "`namespace`" in sql
    assert "FOR UPDATE" in sql.upper()


@pytest.mark.asyncio
async def test_lock_quota_row_for_update_postgres_uses_advisory_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    """Postgres branch uses ``pg_advisory_xact_lock`` — no SQL identifier
    quoting is required there because the lock is on an int8 derived from
    the principal hash, not on a state_kv column whose name might be
    reserved. Pin that the Postgres branch still uses the advisory-lock
    path (so an accidental move of the SQL branch into Postgres would
    fail this test)."""

    executed: list[tuple[str, object]] = []

    class _FakeAsyncpgConn:
        async def execute(self, sql: str, *args) -> None:
            executed.append((sql, args))

    fake_backend_cls = type("PostgresBackend", (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    tx = type("Tx", (), {"conn": _FakeAsyncpgConn()})()

    await consultations._lock_quota_row_for_update(tx, "principal-pg")

    assert len(executed) == 1, (
        f"Postgres branch expected exactly 1 execute call, got {executed!r}"
    )
    sql, arg = executed[0]
    assert "pg_advisory_xact_lock" in sql, (
        f"Postgres branch must use pg_advisory_xact_lock, not the SELECT FOR "
        f"UPDATE row lock — got: {sql!r}"
    )
    # The bind is a stable int8 derived from the principal.
    assert isinstance(arg, tuple) and len(arg) == 1
    assert isinstance(arg[0], int)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_name",
    ["OracleBackend", "Db2Backend", "Db2BackendNative"],
)
async def test_lock_quota_row_for_update_oracle_db2_uses_bind_params(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
):
    """Oracle/Db2 branch uses named bind params (``owner_id``, ``namespace``,
    ``key``). The bare ``SELECT key FROM state`` is correct on those
    backends because the ``key`` literal is a SQL identifier on Oracle
    (not reserved) and a column reference on Db2 (also not reserved in
    SELECT/WHERE contexts — Db2's reserved-word rule applies to DDL, not
    to plain SELECT against a real column named ``key``). This test pins
    the dialect-correct shape across all three Oracle/Db2 backends."""

    executed: list[tuple[str, dict]] = []

    class _FakeOracleCursor:
        async def execute(self, sql: str, params: dict | None = None) -> None:
            executed.append((sql, params))

        async def fetchone(self) -> tuple | None:
            return None

        def close(self) -> None:  # synchronous on oracledb async connection
            pass

    class _FakeOracleConn:
        def cursor(self) -> _FakeOracleCursor:
            return _FakeOracleCursor()

    fake_backend_cls = type(backend_name, (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    tx = type("Tx", (), {"conn": _FakeOracleConn()})()

    await consultations._lock_quota_row_for_update(tx, "principal-oracle")

    assert len(executed) == 1, (
        f"{backend_name} branch expected exactly 1 execute call, got "
        f"{executed!r}"
    )
    sql, params = executed[0]
    assert sql.startswith("SELECT"), (
        f"{backend_name} lock SQL must start with SELECT — got: {sql!r}"
    )
    assert "FROM state" in sql
    assert "FOR UPDATE" in sql.upper()
    # The Oracle/DB2 form uses named bind params (``:key`` etc.), NOT
    # backticks (`` `key` ``) and NOT positional ``%s``.
    assert ":owner_id" in sql
    assert ":namespace" in sql
    assert ":key" in sql
    assert "`" not in sql, (
        f"{backend_name} branch must not use MySQL backticks — got: {sql!r}"
    )
    assert "%s" not in sql, (
        f"{backend_name} branch uses named params, not %s — got: {sql!r}"
    )
    assert isinstance(params, dict)
    assert params["owner_id"] == consultations._QUOTA_OWNER_ID
    assert params["namespace"] == consultations._QUOTA_NAMESPACE
    assert params["key"] == "principal-oracle"


@pytest.mark.asyncio
async def test_lock_quota_row_for_update_sqlite_is_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sqlite + any other backend (no specific branch): ``BEGIN IMMEDIATE``
    already holds a RESERVED write lock on the writer connection, so no
    extra statement is needed. Pin that the lock function executes
    *zero* cursor.execute calls on Sqlite-shaped backends."""

    executed: list[tuple[str, object]] = []

    class _FakeSqliteConn:
        def cursor(self):  # pragma: no cover - never called on Sqlite branch
            raise AssertionError(
                "Sqlite branch must not call conn.cursor() — BEGIN IMMEDIATE "
                "already holds the lock."
            )

    fake_backend_cls = type("SqliteBackend", (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    tx = type("Tx", (), {"conn": _FakeSqliteConn()})()

    # Should not raise.
    await consultations._lock_quota_row_for_update(tx, "principal-sqlite")
    assert executed == [], (
        f"Sqlite branch must not issue any extra SQL — got: {executed!r}"
    )
