import numpy as np
import numpy.typing as npt


def compute_sphere_radius(frac, T, R, R3=None):
    R3 = R**3 if R3 is None else R3
    return np.cbrt(frac*((R + T)**3-R3)+R3)


def compute_tangent_vectors(vectors: npt.NDArray) -> npt.NDArray:
    """Orthogonalize the identity matrix wrt. v and compute SVD to get basis
    for tangent plane.

    Parameters
    ----------
    v : npt.NDArray
        A single vector or an array of vectors (in rows).

    Returns
    -------
    V : npt.NDArray
        Vectors spanning the plane orthogonal to each vector in `v`. If v has
        shape (n, 3) then V has shape (n, 2, 3).
    """
    v = np.atleast_2d(vectors)[..., None]
    I = np.identity(v.shape[1])[None]
    _, S, V = np.linalg.svd(I - I @ v / (np.sum(v**2, axis=1)[:, None]) @ v.transpose(0,2,1))
    assert np.allclose(S[:,-1], 0)
    assert np.allclose(S[:,:-1], 1), "Degenerate elements encountered"

    return V[:,:2].squeeze()
