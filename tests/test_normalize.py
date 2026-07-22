import numpy as np
import pytest

from guardrail_audit.clustering.normalize import cosine_similarity, l2_normalize


def test_l2_normalize_rows_unit_norm():
    m = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = l2_normalize(m)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-9)


def test_l2_normalize_handles_zero_vector():
    out = l2_normalize(np.zeros((1, 4)))
    assert np.all(np.isfinite(out))


def test_cosine_identical_is_one():
    v = np.array([0.2, 0.5, -0.1])
    sim = cosine_similarity(v, v[None, :])
    np.testing.assert_allclose(sim, [1.0], atol=1e-9)


def test_cosine_orthogonal_is_zero():
    q = np.array([1.0, 0.0])
    c = np.array([[0.0, 5.0]])
    np.testing.assert_allclose(cosine_similarity(q, c), [0.0], atol=1e-9)


def test_cosine_normalizes_non_unit_centroids():
    # A centroid that is NOT unit-norm must still give cosine 1 when parallel.
    q = np.array([1.0, 1.0])
    c = np.array([[10.0, 10.0]])
    np.testing.assert_allclose(cosine_similarity(q, c), [1.0], atol=1e-9)
