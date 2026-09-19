"""Fail-closed detection for common prompt-injection and unsafe requests."""

import re


INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions?\b", re.I),
    re.compile(r"\b(disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|system)\b", re.I),
    re.compile(r"\b(bypass|disable|circumvent)\s+(?:security|safety|policy|rules?)\b", re.I),
    re.compile(r"\b(jailbreak|developer\s+message|system\s+prompt)\b", re.I),
    re.compile(r"\b(drop|truncate|alter|create|grant|revoke)\s+(?:table|database|schema|user|view)\b", re.I),
)


def detect_prompt_injection(text: str) -> str | None:
    value = str(text or "")
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(0)
    return None
