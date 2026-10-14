#!/usr/bin/env python3
"""Audit candidate publications and select the production Mesh default."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.validation.tilesets import audit_candidate


def _plan_digest(plan: dict[str, Any]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def select_default(
    plan: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    *,
    publication_bytes: dict[str, int],
    structure: dict[str, dict[str, Any]],
    http: dict[str, dict[str, Any]],
) -> dict[str, object]:
    if plan.get("schemaVersion") != "mesh-candidate-selection.v1":
        raise ValueError("invalid selection plan schema")
    budgets = plan["budgets"]
    candidates: dict[str, dict[str, object]] = {}
    for candidate_id in plan["productionCandidateIds"]:
        profile = profiles[candidate_id]
        if profile.get("schemaVersion") != "mesh-pair-profile.v1":
            raise ValueError(f"profile schema mismatch: {candidate_id}")
        if profile.get("protocolDigest") != plan["comparisonProtocolDigest"]:
            raise ValueError(f"profile protocol mismatch: {candidate_id}")
        if profile.get("models", {}).get("right", {}).get("candidateId") != candidate_id:
            raise ValueError(f"profile candidate mismatch: {candidate_id}")
        if (
            profile.get("viewerBinding", {}).get("models", {}).get("right", {}).get("candidateId")
            != candidate_id
        ):
            raise ValueError(f"profile viewer binding mismatch: {candidate_id}")
        if profile.get("views") != ["overview", "building", "texture", "edge"]:
            raise ValueError(f"profile views mismatch: {candidate_id}")
        phases = profile["phases"]
        visible_triangles = max(int(phase["final"]["right"]["triangles"]) for phase in phases)
        visible_contents = max(int(phase["final"]["right"]["visible"]) for phase in phases)
        peak_memory = max(int(phase["peaks"]["rightMemoryBytes"]) for phase in phases)
        failed_requests = len(profile.get("failedRequests", []))
        console_errors = len(profile.get("errors", []))
        timed_out_views = [phase.get("view") for phase in phases if phase.get("timedOut")]
        results = {
            "publicationBytes": publication_bytes[candidate_id] <= budgets["maxPublicationBytes"],
            "browserTransferBytes": profile["resources"]["right"]["bytes"]
            <= budgets["maxBrowserTransferBytes"],
            "peakMemoryBytes": peak_memory <= budgets["maxPeakMemoryBytes"],
            "visibleTriangles": visible_triangles >= budgets["minVisibleTriangles"],
            "visibleContents": visible_contents >= budgets["minVisibleContents"],
            "failedRequests": failed_requests <= budgets["maxFailedRequests"],
            "consoleErrors": console_errors == 0,
            "structure": bool(structure[candidate_id]["passed"]),
            "httpRangeCors": bool(http[candidate_id]["passed"]),
        }
        profile_passed = not timed_out_views and not profile.get("errors") and failed_requests == 0
        candidates[candidate_id] = {
            "metrics": {
                "publicationBytes": publication_bytes[candidate_id],
                "browserTransferBytes": profile["resources"]["right"]["bytes"],
                "peakMemoryBytes": peak_memory,
                "maxVisibleTriangles": visible_triangles,
                "maxVisibleContents": visible_contents,
                "failedRequests": failed_requests,
                "consoleErrors": console_errors,
            },
            "budgetResults": results,
            "meetsSelectionBudget": all(results.values()),
            "profilePassed": profile_passed,
            "timedOutViews": timed_out_views,
        }

    selected = next(
        (
            candidate_id
            for candidate_id in plan["preferenceOrder"]
            if candidates[candidate_id]["meetsSelectionBudget"]
        ),
        None,
    )
    if selected is None:
        status = "no-candidate-meets-budget"
    elif candidates[selected]["profilePassed"]:
        status = "selected"
    else:
        status = "selected-with-known-timeout-boundary"
    return {
        "schemaVersion": "mesh-candidate-selection-result.v1",
        "selectionPlanDigest": _plan_digest(plan),
        "comparisonProtocolDigest": plan["comparisonProtocolDigest"],
        "defaultCandidateId": selected,
        "selectionStatus": status,
        "timeoutDisposition": plan["timeoutDisposition"],
        "candidates": candidates,
    }


def _assignments(values: list[str], *, json_values: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key or not raw:
            raise ValueError(f"expected ID=VALUE, got {value!r}")
        result[key] = json.loads(Path(raw).read_text(encoding="utf-8")) if json_values else int(raw)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--candidate-id", required=True)
    audit.add_argument("--root", required=True, type=Path)
    audit.add_argument("--base-url", required=True)
    audit.add_argument("--origin", default="http://127.0.0.1:3000")
    audit.add_argument("--report", required=True, type=Path)
    selection = subparsers.add_parser("select")
    selection.add_argument("--plan", required=True, type=Path)
    selection.add_argument("--profile", action="append", default=[])
    selection.add_argument("--audit", action="append", default=[])
    selection.add_argument("--publication-bytes", action="append", default=[])
    selection.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        result = audit_candidate(args.candidate_id, args.root, args.base_url, args.origin)
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        profiles = _assignments(args.profile, json_values=True)
        audits = _assignments(args.audit, json_values=True)
        result = select_default(
            plan,
            profiles,
            publication_bytes=_assignments(args.publication_bytes),
            structure={key: value["filesystem"] for key, value in audits.items()},
            http={key: value["http"] for key, value in audits.items()},
        )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result.get("passed", result.get("defaultCandidateId") is not None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
