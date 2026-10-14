"""Check the commit subject, including Git's temporary fixup subjects."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SUBJECT = re.compile(
    r"(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|release)"
    r"(?:\([^()\r\n]+\))?!?: \S.*"
)


def main() -> int:
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    subject = text.splitlines()[0] if text.splitlines() else ""
    subject = subject.removeprefix("fixup! ")
    if SUBJECT.fullmatch(subject) or subject.lower() == "first commit":
        return 0
    print("Commit subject must use <type>(optional-scope): description", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
