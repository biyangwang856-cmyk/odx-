from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("message", "accepted"),
    [
        ("refactor(core): separate shared processing\n", True),
        ("feat(engine)!: change input contract\n", True),
        ("fixup! feat(engine): add processing\n", True),
        ("release(image): 0.2.0\n", True),
        ("bad subject\n\nfeat: valid-looking body\n", False),
        ("feat: \n", False),
        ("", False),
    ],
)
def test_commit_message_hook_checks_the_subject(tmp_path: Path, message: str, accepted: bool) -> None:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(message)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/quality/check_commit_message.py"), str(path)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == (0 if accepted else 1)
