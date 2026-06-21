from __future__ import annotations

import logging
import os
import sys

LOG = logging.getLogger("mmdt")


def setup_logging() -> None:
    if LOG.handlers:
        return

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    level = logging.DEBUG if os.environ.get("LOG_VERBOSE") else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    LOG.setLevel(level)


def verbose_blocks() -> bool:
    return os.environ.get("LOG_VERBOSE_BLOCKS", "").lower() in ("1", "true", "yes")
