"""Allow ``python -m jev_course ...`` alongside the ``jev-course`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
