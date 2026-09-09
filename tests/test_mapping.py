"""Score, distance, fusion."""

import pytest
from mistralai.search.toolkit.embedding import DistanceMetric

from mistralai.search.toolkit.plugins.qdrant.mapping import (
    distance_to_score,
    fuse,
    score_to_distance,
)


def _payload(cid: str) -> dict:
    return {
        "id": cid,
        "source_id": "s.md",
        "locator": f"char:{cid}-{cid}",
        "start_offset": 0,
        "end_offset": 10,
        "chunk_type": "content",
        "content": "hello",
        "metadata": {},
    }


@pytest.mark.parametrize(
    ("metric", "raw_score", "expected_distance"),
    [
        (DistanceMetric.COSINE, 1.0, 0.0),
        (DistanceMetric.COSINE, 0.7071, pytest.approx(0.2929, abs=1e-4)),
        (DistanceMetric.COSINE, 0.0, 1.0),
        (DistanceMetric.L2, 0.0, 0.0),
        (DistanceMetric.L2, 1.4142, pytest.approx(1.4142, abs=1e-4)),
        (DistanceMetric.INNER_PRODUCT, 1.0, -1.0),
        (DistanceMetric.INNER_PRODUCT, 0.0, 0.0),
    ],
)
def test_score_to_distance(metric, raw_score, expected_distance):
    assert score_to_distance(metric, raw_score) == expected_distance


@pytest.mark.parametrize(
    "metric", [DistanceMetric.COSINE, DistanceMetric.INNER_PRODUCT]
)
def test_score_distance_round_trip(metric):
    for distance in (0.0, 0.5, 2.0):
        assert score_to_distance(
            metric, distance_to_score(metric, distance)
        ) == pytest.approx(distance)


def test_fused_results_distances():
    payloads = {cid: _payload(cid) for cid in ("a", "b", "c")}
    results = fuse(
        [("a", 0.9), ("b", 0.5)],
        ["b", "c"],
        payloads,
        DistanceMetric.COSINE,
        (),
        top_k=3,
        vector_weight=1.0,
        text_weight=1.0,
        rrf_k=4,
        include_content=True,
        include_metadata=True,
    )
    assert [r.chunk.id for r in results] == ["b", "a", "c"]
    by_id = {r.chunk.id: r for r in results}
    assert by_id["a"].distance == pytest.approx(0.1)
    assert by_id["b"].distance == pytest.approx(0.5)
    assert by_id["c"].distance is None
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert by_id["b"].score == pytest.approx(1 / 6 + 1 / 5)
