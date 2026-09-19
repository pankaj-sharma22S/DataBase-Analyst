"""Sensitive-data detection and masking. Salary is intentionally not sensitive."""

import ast
import json
import re
from typing import Any


SECRET_FIELD = re.compile(
    r"(?:password|passwd|pin|cvv|cvc|otp|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|secret|private[_ -]?key|authorization)",
    re.I,
)
BANK_FIELD = re.compile(r"(?:bank[_ -]?account|account[_ -]?number|iban)", re.I)
SECRET_VALUE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{12,}|Bearer\s+[A-Za-z0-9._~+/=-]{12,})\b")


def is_sensitive_field(field: Any) -> bool:
    return bool(SECRET_FIELD.search(str(field)))


def mask_sensitive_value(value: Any, field: Any = "") -> Any:
    if value is None:
        return value
    if is_sensitive_field(field):
        return "[REDACTED]"
    if BANK_FIELD.search(str(field)):
        digits = re.sub(r"\D", "", str(value))
        return "********" + digits[-4:] if len(digits) >= 4 else "[REDACTED]"
    if isinstance(value, str):
        return SECRET_VALUE.sub("[REDACTED]", value)
    return value


def mask_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: mask_sensitive_value(value, key) for key, value in record.items()}


def mask_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [mask_record(row) for row in rows]


def mask_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(
        r"(?i)(password|passwd|pin|cvv|cvc|otp|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|secret|private[_ -]?key)\s*[:=]\s*([^,;\s]+)",
        lambda match: f"{match.group(1)}=[REDACTED]",
        value,
    )
    def _mask_account(match):
        digits = re.sub(r"\D", "", match.group(2))
        return f"{match.group(1)}=********{digits[-4:]}"

    value = re.sub(
        r"(?i)(bank[_ -]?account|account[_ -]?number|iban)\s*[:=]\s*([0-9A-Z -]{4,})",
        _mask_account,
        value,
    )
    return SECRET_VALUE.sub("[REDACTED]", value)


def bounded_rows(rows: list[dict[str, Any]], max_rows: int, max_bytes: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        candidate = mask_record(row)
        if len(json.dumps(result + [candidate], default=str)) > max_bytes:
            break
        result.append(candidate)
    return result


def mask_tool_result(result: Any) -> Any:
    """Mask SQL-tool output before it becomes an agent observation."""
    if not isinstance(result, str):
        return result
    try:
        parsed = ast.literal_eval(result)
    except (SyntaxError, ValueError):
        return mask_text(result)
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return str([mask_record(item) for item in parsed])
    return mask_text(result)
