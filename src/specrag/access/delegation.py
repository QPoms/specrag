"""Short-lived signed service delegation for the internal MCP boundary."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets

from specrag.models import ActorContext, ServiceDelegation

from .policy import AuthorizationError


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(value + padding)
    except (ValueError, UnicodeEncodeError) as exc:
        raise AuthorizationError("invalid service delegation") from exc
    if not value or _encode(decoded) != value:
        raise AuthorizationError("invalid service delegation")
    return decoded


class ServiceDelegationSigner:
    def __init__(
        self,
        secret: bytes,
        *,
        max_lifetime: timedelta = timedelta(minutes=5),
        clock_skew: timedelta = timedelta(seconds=10),
    ) -> None:
        if len(secret) < 32:
            raise ValueError("delegation secret must contain at least 32 bytes")
        self.secret = secret
        self.max_lifetime = max_lifetime
        self.clock_skew = clock_skew

    def issue(
        self,
        context: ActorContext,
        *,
        conversation_id: str,
        origin_message_id: str,
        origin_turn_id: str,
        submitted_user_sequence: int,
        tool_call_id: str,
        operation_key: str,
        lifetime: timedelta = timedelta(seconds=60),
        now: datetime | None = None,
    ) -> str:
        now = now or datetime.now(timezone.utc)
        if lifetime <= timedelta(0) or lifetime > self.max_lifetime:
            raise ValueError("delegation lifetime is outside the allowed range")
        expires_at = min(now + lifetime, context.expires_at)
        delegation = ServiceDelegation(
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            conversation_id=conversation_id,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            submitted_user_sequence=submitted_user_sequence,
            tool_call_id=tool_call_id,
            operation_key=operation_key,
            issued_at=now,
            expires_at=expires_at,
            nonce=secrets.token_urlsafe(18),
        )
        payload = json.dumps(
            delegation.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.new(self.secret, payload, hashlib.sha256).digest()
        return _encode(payload) + "." + _encode(signature)

    def verify(
        self, token: str, *, now: datetime | None = None
    ) -> ServiceDelegation:
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
        except ValueError as exc:
            raise AuthorizationError("invalid service delegation") from exc
        payload = _decode(encoded_payload)
        supplied = _decode(encoded_signature)
        expected = hmac.new(self.secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            raise AuthorizationError("invalid service delegation")
        try:
            delegation = ServiceDelegation.model_validate_json(payload)
        except ValueError as exc:
            raise AuthorizationError("invalid service delegation") from exc
        current = now or datetime.now(timezone.utc)
        if delegation.issued_at > current + self.clock_skew:
            raise AuthorizationError("invalid service delegation")
        if delegation.expires_at <= current - self.clock_skew:
            raise AuthorizationError("expired service delegation")
        if delegation.expires_at - delegation.issued_at > self.max_lifetime:
            raise AuthorizationError("invalid service delegation")
        return delegation


__all__ = ["ServiceDelegationSigner"]
