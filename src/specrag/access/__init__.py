"""Framework-independent authorization policy."""

from .policy import AccessPolicy, AuthorizationError, Permission
from .service import IdentityStore, LocalIdentityService

__all__ = [
    "AccessPolicy", "AuthorizationError", "IdentityStore",
    "LocalIdentityService", "Permission",
]
