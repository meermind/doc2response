import logging
import os
from typing import Optional

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except Exception:
    RichHandler = None  # type: ignore
    _HAS_RICH = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure global logging with colorized output if available.

    Idempotent: safe to call multiple times.
    Level can be overridden via param or env var D2R_LOG_LEVEL.
    """
    root = logging.getLogger()
    if getattr(root, "_d2r_configured", False):
        return

    level_name = (level or os.getenv("D2R_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    # Remove default handlers if any basicConfig was called elsewhere
    for h in list(root.handlers):
        root.removeHandler(h)

    if _HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False, markup=True)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    logging.basicConfig(level=numeric_level, format=fmt, handlers=[handler])
    root._d2r_configured = True  # type: ignore[attr-defined]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)


