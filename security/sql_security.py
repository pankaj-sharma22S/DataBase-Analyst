"""Read-only SQL validation and guarded database proxy."""

import re
from typing import Any

from langchain_community.utilities.sql_database import SQLDatabase
from .pii_detector import mask_tool_result


class SQLSecurityError(ValueError):
    pass


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|merge|grant|revoke|attach|detach|pragma|vacuum|commit|rollback)\b",
    re.I,
)
READ_START = re.compile(r"^(select|with|show|describe|desc|explain)\b", re.I)


def _without_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`", "''", sql)


def statement_count(sql: str) -> int:
    count = 0
    has_content = False
    quote = None
    i = 0
    while i < len(sql):
        char = sql[i]
        if quote:
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 1
                else:
                    quote = None
        elif char in "'\"`":
            quote = char
        elif char == ";":
            if has_content:
                count += 1
                has_content = False
        elif not char.isspace():
            has_content = True
        i += 1
    if has_content:
        count += 1
    return count


def validate_read_only_sql(sql: str) -> str:
    query = str(sql or "").strip()
    if not query:
        raise SQLSecurityError("Security policy blocked an empty SQL query.")
    if statement_count(query) > 1:
        raise SQLSecurityError("Security policy blocked multiple SQL statements.")
    normalized = query.rstrip().rstrip(";").strip()
    if not READ_START.match(normalized):
        raise SQLSecurityError("Security policy allows read-only SQL queries only.")
    if FORBIDDEN_SQL.search(_without_literals(normalized)):
        raise SQLSecurityError("Security policy blocked a destructive SQL operation.")
    return normalized


class ReadOnlyDatabaseProxy(SQLDatabase):
    """Delegates schema methods while guarding every toolkit SQL execution."""

    def __init__(self, database: Any, validator, timeout_seconds: float | None = None):
        if not isinstance(database, SQLDatabase):
            raise TypeError("ReadOnlyDatabaseProxy requires an SQLDatabase instance.")
        self.__dict__ = database.__dict__.copy()
        self._validator = validator
        self._timeout_seconds = timeout_seconds

    def run(self, command, *args, **kwargs):
        self._validator(str(command))
        kwargs = dict(kwargs)
        kwargs["include_columns"] = True
        operation = lambda: mask_tool_result(super(ReadOnlyDatabaseProxy, self).run(command, *args, **kwargs))
        return operation() if self._timeout_seconds is None else self._run_with_timeout(operation)

    def run_no_throw(self, command, *args, **kwargs):
        self._validator(str(command))
        kwargs = dict(kwargs)
        kwargs["include_columns"] = True
        operation = lambda: mask_tool_result(super(ReadOnlyDatabaseProxy, self).run_no_throw(command, *args, **kwargs))
        return operation() if self._timeout_seconds is None else self._run_with_timeout(operation)

    def _run_with_timeout(self, operation):
        from .security_gateway import security_gateway
        return security_gateway.run_with_timeout(operation, self._timeout_seconds)
