from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from typing import Dict, Optional

from .config import Config

_PBKDF2_ITERS = 200_000


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS).hex()


class Auth:
    """Minimal role-based auth (admin/teacher) in users.json. Stdlib only."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.path = os.path.join(cfg.db_dir, "users.json")
        self._lock = threading.RLock()
        self.default_admin = not self._has_admin()
        if self.default_admin:  # first run — bootstrap from env/secrets
            self.add_user(os.environ.get("ATT_ADMIN_USER", "admin"),
                          os.environ.get("ATT_ADMIN_PASSWORD", "admin123"),
                          role="admin")

    # ---------- storage ----------
    def _load(self) -> Dict[str, dict]:
        with self._lock:
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            return {}

    def _save(self, users: Dict[str, dict]) -> None:
        with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.path)

    def _has_admin(self) -> bool:
        return any(u.get("role") == "admin" for u in self._load().values())

    # ---------- API ----------
    def verify(self, username: str, password: str) -> Optional[dict]:
        u = self._load().get(username.strip().lower())
        time.sleep(0.7)  # uniform delay: brute-force throttle + no user enumeration
        if not u or _hash(password, bytes.fromhex(u["salt"])) != u["hash"]:
            return None
        return {"username": username.strip().lower(), "role": u["role"]}

    def add_user(self, username: str, password: str, role: str = "teacher") -> bool:
        username = username.strip().lower()
        if not username or len(password) < 6 or role not in ("admin", "teacher"):
            return False
        users = self._load()
        if username in users:
            return False
        salt = secrets.token_bytes(16)
        users[username] = {"role": role, "salt": salt.hex(),
                           "hash": _hash(password, salt),
                           "created": self.cfg.now().isoformat(timespec="seconds")}
        self._save(users)
        return True

    def set_password(self, username: str, new_password: str) -> bool:
        users = self._load()
        if username not in users or len(new_password) < 6:
            return False
        salt = secrets.token_bytes(16)
        users[username].update(salt=salt.hex(), hash=_hash(new_password, salt))
        self._save(users)
        return True

    def remove_user(self, username: str) -> bool:
        users = self._load()
        if username not in users:
            return False
        admins = sum(1 for u in users.values() if u["role"] == "admin")
        if users[username]["role"] == "admin" and admins <= 1:
            return False  # never remove the last admin
        users.pop(username)
        self._save(users)
        return True

    def list_users(self, role: Optional[str] = None) -> Dict[str, str]:
        return {k: v["role"] for k, v in self._load().items()
                if role in (None, v["role"])}
