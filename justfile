set shell := ["bash", "-e", "-o", "pipefail", "-c"]

default:
    @just --list

sync:
    pixi install --environment default
    pixi install --environment viewer
    pixi run --environment viewer npm --prefix services/viewer ci
    pixi run --environment default npm --prefix browser/playwright ci

list:
    pixi run --environment default python -m experiments.cli list

check:
    pixi run --environment viewer npm --prefix services/viewer run check
    pixi run --environment viewer npm --prefix services/viewer test
    ./scripts/quality/check-python.sh
    pixi run --environment viewer python -m pytest services/viewer_api/tests
    pixi run --environment default node --test tests/test_gpu_textures.cjs
    pixi run --environment default node --check browser/playwright/audit_visible_tiles.cjs
    pixi run --environment default node --check browser/playwright/profile_interactive_path.cjs
    pixi run --environment default node --check browser/playwright/profile_pair_visual.cjs
    pixi run --environment default node --test browser/playwright/profile_contract.test.cjs

check-python:
    ./scripts/quality/check-python.sh

install-hooks:
    devbox run -- lefthook install

changelog:
    devbox run -- git-cliff --config cliff.toml --output CHANGELOG.md

test:
    pixi run --environment default python -m pytest tests
    pixi run --environment viewer python -m pytest services/viewer_api/tests

format:
    pixi run --environment default ruff format app tests services/viewer_api
    pixi run --environment default ruff check --fix app tests services/viewer_api

task name *args:
    pixi run --environment default python -m experiments.cli run {{ name }} {{ args }}

remesh-runtime *args:
    pixi run --environment default python -m experiments.cli run build.remesh-runtime {{ args }}

viewer-build:
    pixi run --environment viewer npm --prefix services/viewer run build

viewer-check:
    pixi run --environment viewer npm --prefix services/viewer run check
    pixi run --environment viewer npm --prefix services/viewer test
    pixi run --environment viewer python -m pytest services/viewer_api/tests

viewer target="v55": viewer-build
    pixi run --environment viewer python -m services.viewer_api.app --target {{ target }}

release-contract:
    bash scripts/ci/test-release-images-rootless.sh

prepare-release version:
    ./scripts/prepare_release.sh {{ quote(version) }}
