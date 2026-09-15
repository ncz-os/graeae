"""Real first-use quota contention; only disposable schemas are permitted."""

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4
from urllib.parse import urlsplit, urlunsplit

import pytest
from fastapi import HTTPException

from mnemos.api.routes import consultations


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["oracle", "db2", "mysql", "mariadb"])
async def test_first_use_quota_serializes(monkeypatch, dialect):
    dsn = os.getenv("MNEMOS_TEST_" + dialect.upper() + "_DSN")
    if not dsn or os.getenv("MNEMOS_TEST_ALLOW_RESET") != "1":
        pytest.skip("dedicated test database required")
    settings = SimpleNamespace(database=SimpleNamespace(embedding_dim=3))
    admin = None
    database = None
    if dialect == "oracle":
        pytest.importorskip("oracledb")
        from mnemos.persistence.oracle import OracleBackend, create_oracle_pool

        backend = OracleBackend(
            await create_oracle_pool(dsn, min_size=2, max_size=4, settings=settings),
            settings,
        )
    elif dialect == "db2":
        pytest.importorskip("ibm_db")
        from mnemos.persistence.db2 import Db2BackendNative, create_db2_native_pool

        backend = Db2BackendNative(
            await create_db2_native_pool(dsn, min_size=2, max_size=4), settings
        )
    else:
        aiomysql = pytest.importorskip("aiomysql")
        from mnemos.persistence.mysql import MysqlBackend, create_mysql_pool
        from mnemos.persistence.mariadb import MariadbBackend, create_mariadb_pool

        parsed = urlsplit(dsn)
        admin = await aiomysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password or "",
            autocommit=True,
        )
        database = "graeae_quota_" + uuid4().hex[:16]
        async with admin.cursor() as cursor:
            await cursor.execute(
                f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        factory, cls = (
            (create_mysql_pool, MysqlBackend)
            if dialect == "mysql"
            else (create_mariadb_pool, MariadbBackend)
        )
        pool = await factory(
            urlunsplit(parsed._replace(path="/" + database)),
            min_size=2,
            max_size=4,
            settings=settings,
        )
        backend = cls(pool, settings)
    principal = "quota-review-" + uuid4().hex
    key = consultations._quota_principal_key(principal)
    monkeypatch.setattr(consultations._lc, "_persistence_backend", backend)
    monkeypatch.setattr(consultations, "backend_or_503", lambda: backend)
    monkeypatch.setattr(consultations, "_consult_quota_per_min", lambda: 1.0)
    try:
        await backend.open()

        async def attempt():
            try:
                await consultations._enforce_consult_quota(principal, "single")
                return 200
            except HTTPException as exc:
                return exc.status_code

        results = await asyncio.gather(*(attempt() for _ in range(8)))
        assert sorted(results) == [200] + [429] * 7
    finally:
        try:
            if admin is None:
                from mnemos.persistence.worker_lifecycle import _Ops

                async with backend.transactional() as tx:
                    await _Ops(tx, dialect).execute(
                        "DELETE FROM state WHERE owner_id=? AND namespace=? AND key=?",
                        consultations._QUOTA_OWNER_ID,
                        consultations._QUOTA_NAMESPACE,
                        key,
                    )
        finally:
            await backend.close()
            if admin is not None:
                try:
                    async with admin.cursor() as cursor:
                        await cursor.execute(f"DROP DATABASE `{database}`")
                finally:
                    admin.close()
