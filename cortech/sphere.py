import numpy as np

from cortech.cgal.convex_hull_3 import convex_hull

def fibonacci_points(n_points: int):
    """
    http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

    see fig. 1 for choice of 'best' epsilon (maximizer of minimum distance)

    """

    if n_points < 30:
        epsilon = 0.
    elif n_points < 150:
        epsilon = 2.
    else:
        epsilon = 2.5

    phi = 0.5 * (1 + np.sqrt(5)) # the golden ratio
    i = np.arange(0, n_points, dtype=float)

    # Fibonacci grid
    x2 = (i + 0.5 + epsilon) / (n_points + 2*epsilon)
    y2 = i * phi
    if n_points >= 30:
        x2[0], y2[0] = 0, 0
        x2[-1], y2[-1] = 1, 0

    # Fibonacci sphere
    # phi   : latitude (from pole to pole, 0 <= phi <= pi)
    # theta : longitude (around sphere, 0 <= theta <= 2*pi)

    # spherical coordinates (r = 1 is implicit because it is unit sphere)
    phi = np.arccos(1 - 2*x2)
    theta = 2*np.pi*y2

    # cartesian coordinates
    x3 = np.cos(theta) * np.sin(phi)
    y3 = np.sin(theta) * np.sin(phi)
    z3 = np.cos(phi)

    return np.array([x3,y3,z3]).T

def fibonacci_sphere(n_points: int, radius: float = 1.0):
    """Generates a triangulated sphere with N vertices and radius r centered on
    (0,0,0).
    """
    pts = fibonacci_points(n_points) * radius
    v, f = convex_hull(pts)
    # v, f = hull.points, hull.simplices
    # orientation_consistency(pts, tri)
    return v, f
