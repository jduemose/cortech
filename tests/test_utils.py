import numpy as np
import pytest

import cortech.utils

@pytest.mark.parametrize("arr", [np.array(1.0), np.array([[1.0]])])
@pytest.mark.parametrize("n", [2,4])
def test_atleast_nd(arr, n):
    x = cortech.utils.atleast_nd(arr, n)
    assert x.ndim == n
    assert x.squeeze().ndim == arr.ndim

def test_sliced_argmin():
    arr = np.array([0,1,2,2,1,0,2,2,2,3,1])
    indptr = [0, 3, 6, 8, len(arr)]
    x = cortech.utils.sliced_argmin(arr, indptr)
    np.testing.assert_allclose(x, [0, 5, 6, 10])


def test_normalize():
    np.random.seed(0)
    a = np.random.randn(10,3)

    b = cortech.utils.normalize(a, axis=0)
    c = cortech.utils.normalize(a, axis=1)

    np.testing.assert_allclose(np.linalg.norm(b, axis=0), 1.0)
    np.testing.assert_allclose(np.linalg.norm(c, axis=1), 1.0)

# def test_compute_sphere_radius():


def test_compute_tangent_vectors():
    v = np.array([[1,0,0], [0,1,0], [0,0,1]])
    u = cortech.utils.compute_tangent_vectors(v)

    # Test that u0 and u1 are both orthogonal to v
    np.testing.assert_allclose(u @ v[:, None].swapaxes(1,2), 0)
    # Test that u0 is orthogonal to u1
    np.testing.assert_allclose(np.sum(u[:,0] * u[:,1], axis=1), 0)

# def test_k_ring_neighbors():

