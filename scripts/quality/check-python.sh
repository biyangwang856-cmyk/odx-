#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
pixi run --environment default ruff check app tests scripts/quality services/viewer_api
pixi run --environment default python -m pytest tests
