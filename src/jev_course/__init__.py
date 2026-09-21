"""Course helpers around the vendored ``jevlike`` package.

This package adds no model, loss or training code. It adds the parts the lesson
needs beyond upstream: disjoint split handling, an independent temperature
calibration with input validation, a self-check command and a full-flow runner
that executes the documented commands as checked subprocesses.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
