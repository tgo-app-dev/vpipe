"""vpipe -- Python entry point.

Importing this package loads the C++ core (`_vpipe`) and creates one
default Session through the SessionManager singleton.

Startup config is resolved in this order (first match wins):

  1. ``VPIPE_CONFIG``  -- environment variable, value passed verbatim
                          to ``SessionManager.create_session``. The C++
                          parser auto-detects:
                            * empty / whitespace -> built-in defaults
                            * starts with '{' or '[' -> inline JSON
                            * otherwise -> filesystem path (JSON or
                              binary FlexData)
  2. ``VPIPE_CONFIG_FILE`` -- environment variable holding a path.
                              Useful when you want to be explicit and
                              avoid the auto-detection in (1).
  3. ``./init.vpipe``  -- file in the current working directory.
  4. built-in defaults (empty config string).

The chosen value is exposed as ``vpipe.config_source`` for debugging.
The created session is ``vpipe.session``; the manager is
``vpipe.manager``. Users that want their own session can call
``vpipe.create_session(config="...")`` directly.
"""

from __future__ import annotations

import atexit
import os

from . import _vpipe
from ._vpipe import (
    PipelineHandle,
    SessionIntf,
    SessionManager,
    StageHandle,
    Status,
    vpipe_version,
)

__all__ = [
    "PipelineHandle",
    "SessionIntf",
    "SessionManager",
    "StageHandle",
    "Status",
    "config_source",
    "create_session",
    "destroy_session",
    "manager",
    "session",
    "vpipe_version",
]


_INIT_FILE_NAME = "init.vpipe"


def _resolve_startup_config() -> tuple[str, str]:
    """Return (config_string, source_label)."""
    val = os.environ.get("VPIPE_CONFIG")
    if val is not None:
        return val, "env:VPIPE_CONFIG"

    path = os.environ.get("VPIPE_CONFIG_FILE")
    if path:
        return path, "env:VPIPE_CONFIG_FILE"

    cwd_init = os.path.join(os.getcwd(), _INIT_FILE_NAME)
    if os.path.isfile(cwd_init):
        return cwd_init, f"cwd:{_INIT_FILE_NAME}"

    return "", "defaults"


manager: SessionManager = SessionManager.get()


def create_session(config: str = "") -> SessionIntf:
    """Create a new Session through the global SessionManager."""
    return manager.create_session(config)


def destroy_session(s: SessionIntf) -> None:
    """Destroy a session previously created through SessionManager."""
    manager.destroy_session(s)


_cfg, config_source = _resolve_startup_config()
session: SessionIntf = manager.create_session(_cfg)
del _cfg


def _shutdown() -> None:
    # Best-effort: tear down the default session at interpreter exit
    # so the underlying ThreadPool / log delegate get cleaned up
    # before Python finalizes module globals.
    global session
    s = session
    session = None  # type: ignore[assignment]
    if s is not None:
        try:
            manager.destroy_session(s)
        except Exception:
            pass


atexit.register(_shutdown)
