import numpy as np
import pytest

from cortech.sphere import fibonacci_points, fibonacci_sphere


@pytest.fixture(scope="module")
def sphere_points(n=100, r=1.0):
    return fibonacci_points(n, r)

@pytest.fixture(scope="module")
def sphere(n=100, r=1.0):
    return fibonacci_sphere(n, r)

@pytest.fixture(scope="module")
def diamond():
    # axis-aligned diamond shape
    vertices = np.array([[0,0,1], [-1,0,0], [0, -1, 0], [1,0,0], [0,1,0], [0,0,-1]])
    faces = np.array([[0,1,2], [0,2,3], [0,3,4], [0,4,1], [1,5,2], [2,5,3], [3,5,4],[4,5,1]])
    return vertices, faces

@pytest.fixture(scope="module")
def diamond_barycenters():
    return np.array([[-0.33333333,-0.33333333,0.33333333],
    [0.33333333,-0.33333333,0.33333333],
    [0.33333333,0.33333333,0.33333333],
    [-0.33333333,0.33333333,0.33333333],
    [-0.33333333,-0.33333333,-0.33333333],
    [0.33333333,-0.33333333,-0.33333333],
    [0.33333333,0.33333333,-0.33333333],
    [-0.33333333,0.33333333,-0.33333333]])

@pytest.fixture(scope="module")
def diamond_adjacency_matrix():
    return np.array(
        [
            [0,1,1,1,1,0], [1,0,1,0,1,1], [1,1,0,1,0,1],
            [1,0,1,0,1,1], [1,1,0,1,0,1], [0,1,1,1,1,0]
        ], dtype=float
    )
