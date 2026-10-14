import json

from experiments.tools.reschedule_hlod_errors import reschedule_shard


def test_original_metric_uses_matching_tile_error(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    original_path = tmp_path / "original.json"
    report_path = tmp_path / "report.json"
    candidate_path.write_text(json.dumps({"root": {"geometricError": 9.0, "content": {"uri": "proxy.glb"}}}))
    original_path.write_text(json.dumps({"root": {"geometricError": 0.1, "content": {"uri": "source.glb"}}}))
    report_path.write_text(
        json.dumps(
            {
                "shard": "0000",
                "nodes": [
                    {
                        "proxy": "proxy.glb",
                        "accepted": True,
                        "measuredError": {"maximumM": 5.0, "areaWeightedP99M": 4.0},
                    }
                ],
            }
        )
    )

    result = reschedule_shard(
        candidate_path,
        report_path,
        "original",
        safety_factor=2.0,
        cap_m=0.15,
        original_path=original_path,
    )

    candidate = json.loads(candidate_path.read_text())
    assert candidate["root"]["geometricError"] == 0.15
    assert result["minimumOwnErrorM"] == 0.15
    assert result["maximumOwnErrorM"] == 0.15
