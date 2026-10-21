"""Configuration loading: merges .env secrets with config/settings.yaml.

Nothing here ever logs a secret. Secrets come only from the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"

load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Secrets:
    trading_mode: str = field(default_factory=lambda: _env("TRADING_MODE", "paper").lower())
    kite_api_key: str = field(default_factory=lambda: _env("KITE_API_KEY"))
    kite_api_secret: str = field(default_factory=lambda: _env("KITE_API_SECRET"))
    kite_access_token: str = field(default_factory=lambda: _env("KITE_ACCESS_TOKEN"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    gemini_max_calls_per_day: int = field(default_factory=lambda: _env_int("GEMINI_MAX_CALLS_PER_DAY", 50))
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: _env("SMTP_USER"))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD"))
    alert_email_to: str = field(default_factory=lambda: _env("ALERT_EMAIL_TO"))

    @property
    def is_live(self) -> bool:
        return self.trading_mode == "live"

    @property
    def has_kite(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)


class Config:
    """Loaded YAML settings + secrets. Access settings via dotted helpers."""

    def __init__(self, settings: dict[str, Any], secrets: Secrets):
        self.settings = settings
        self.secrets = secrets
        DATA_DIR.mkdir(exist_ok=True)
        LOG_DIR.mkdir(exist_ok=True)

    # -- convenience accessors --
    @property
    def risk(self) -> dict[str, Any]:
        return self.settings.get("risk", {})

    @property
    def session(self) -> dict[str, Any]:
        return self.settings.get("session", {})

    @property
    def account(self) -> dict[str, Any]:
        return self.settings.get("account", {})

    @property
    def strategies(self) -> list[dict[str, Any]]:
        return self.settings.get("strategies", [])

    @property
    def ai(self) -> dict[str, Any]:
        return self.settings.get("ai", {})

    def enabled_strategies(self) -> list[dict[str, Any]]:
        return [s for s in self.strategies if s.get("enabled")]

    def save(self) -> None:
        """Persist settings back to YAML (used by the agent after approval)."""
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.settings, fh, sort_keys=False, allow_unicode=True)


def load_config() -> Config:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh) or {}
    else:
        settings = {}
    return Config(settings=settings, secrets=Secrets())
