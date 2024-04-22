import numpy as np
import numpy.typing as npt


def sliced_argmin(x: npt.NDArray, indptr: npt.NDArray):
    """Perform argmin on slices of x.

    PARAMETERS
    ----------
    x : 1-d array
        The array to perform argmin on.
    indptr : 1-d array-like
        The indices of the slices. The ith slice is indptr[i]:indptr[i+1].

    RETURNS
    -------
    res : 1-d array
        The indices (into x) corresponding to the minimum values in each chunk.
    """
    assert x.ndim == 1
    return np.array([x[i:j].argmin() + i for i, j in zip(indptr[:-1], indptr[1:])])


def normalize(arr: npt.NDArray, axis=None, inplace: bool = False):
    """Normalize along a particular axis (or axes) of `v` avoiding
    RuntimeWarning due to division by zero.

    PARAMETERS
    ----------
    v : ndarray
        Array to nomalize.
    axis:
        The axis to normalize along, e.g., `axis=1` will normalize rows).
        Like axis argument of numpy.linalg.norm.

    RETURNS
    -------
    v (normalized)
    """
    size = np.linalg.norm(arr, axis=axis, keepdims=True)
    if inplace:
        np.divide(arr, size, where=size != 0, out=arr)
    else:
        return np.divide(arr, size, where=size != 0)


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
