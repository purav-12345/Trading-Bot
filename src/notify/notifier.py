"""Alerts via Telegram and/or email. Both are optional; if not configured
the notifier just logs to stdout so the bot still runs everywhere."""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

import requests


class Notifier:
    def __init__(self, config):
        self.config = config
        s = config.secrets
        self.tg_token = s.telegram_bot_token
        self.tg_chat = s.telegram_chat_id
        self.smtp_host = s.smtp_host
        self.smtp_port = s.smtp_port
        self.smtp_user = s.smtp_user
        self.smtp_pass = s.smtp_password
        self.email_to = s.alert_email_to

    def send(self, message: str, subject: str = "TradingBot Alert") -> None:
        print(f"[ALERT] {message}")
        if self.tg_token and self.tg_chat:
            self._telegram(message)
        if self.smtp_host and self.email_to:
            self._email(subject, message)

    def _telegram(self, message: str) -> None:
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                json={"chat_id": self.tg_chat, "text": message},
                timeout=10,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[notify] telegram failed: {exc}")

    def _email(self, subject: str, message: str) -> None:
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.smtp_user or "tradingbot"
            msg["To"] = self.email_to
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.starttls()
                if self.smtp_user:
                    server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"[notify] email failed: {exc}")
