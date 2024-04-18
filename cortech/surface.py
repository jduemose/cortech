import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import scipy.sparse
from scipy.ndimage import map_coordinates

import cortech.utils

from cortech.constants import Curvature


class Surface:
    def __init__(
        self, vertices: npt.NDArray, faces: npt.NDArray, metadata, initialize=True
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.metadata = metadata

        if initialize:
            self.vertex_adjacency = self.compute_vertex_adjacency()

    def compute_vertex_adjacency(self, with_diag=False):
        """Make sparse adjacency matrix for vertices with connections `tris`."""
        N = self.faces.max() + 1

        pairs = list(itertools.combinations(np.arange(self.faces.shape[1]), 2))
        row_ind = np.concatenate([self.faces[:, i] for p in pairs for i in p])
        col_ind = np.concatenate([self.faces[:, i] for p in pairs for i in p[::-1]])

        data = np.ones_like(row_ind)
        A = scipy.sparse.csr_array((data / 2, (row_ind, col_ind)), shape=(N, N))

        if with_diag:
            A = A.tolil()
            A.setdiag(1)
            A = A.tocsr()

        return A

    def compute_face_normals(self):
        """Get normal vectors for each triangle in the mesh.

        PARAMETERS
        ----------
        mesh : ndarray
            Array describing the surface mesh. The dimension are:
            [# of triangles] x [vertices (of triangle)] x [coordinates (of vertices)].

        RETURNS
        ----------
        tnormals : ndarray
            Normal vectors of each triangle in "mesh".
        """
        mesh = self.vertices[self.faces]

        tnormals = np.cross(
            mesh[:, 1, :] - mesh[:, 0, :], mesh[:, 2, :] - mesh[:, 0, :]
        ).astype(float)
        tnormals /= np.sqrt(np.sum(tnormals**2, 1))[:, np.newaxis]

        return tnormals

    def compute_vertex_normals(self):
        """ """
        face_normals = self.compute_face_normals()

        out = np.zeros_like(self.vertices)
        for i in range(len(self.faces)):
            out[self.faces[i]] += face_normals[i]
        out /= np.linalg.norm(out, ord=2, axis=1)[:, None]

        return out

    def compute_principal_curvatures(self):
        """Compute principal curvatures and corresponding directions. From these,
        the following curvature estimates can easily be calculated

        Mean curvature

            H = 0.5*(k1+k2)

        Gaussian curvature

            K = k1*k2


        Parameters
        ----------
        v : npt.NDArray
            Vertices
        f : npt.NDArray
            Faces

        Returns
        -------
        D : ndarray
            Principal curvatures with k1 and k2 (maximum and minimum curvature,
            respectively) in first and second column.
        E : ndarray
            Principal directions corresponding to the principal curvatures.

        Notes
        -----
        This function is similar to Freesurfer's
        `MRIScomputeSecondFundamentalForm`.
        """
        n = self.vertices.shape[0]

        adj = self.compute_vertex_adjacency()
        vn = self.compute_vertex_normals()
        vt = cortech.utils.compute_tangent_vectors(vn)

        m = np.array(adj.sum(1)).squeeze().astype(int)  # number of neighbors
        muq = np.unique(m)

        # Estimate the parameters of the second fundamental form at each vertex.
        # The second fundamental form is a quadratic form on the tangent plane of
        # the vertex
        # (see https://en.wikipedia.org/wiki/Second_fundamental_form)

        # We cannot solve for all vertices at the same time as the number of
        # equations in the system equals the number of neighbors. However, we can
        # solve all vertices with the same number of neighbors concurrently as this
        # is broadcastable

        H_uv = np.zeros((n, 2, 2))
        for mm in muq:
            i = np.where(m == mm)[0]
            vi = self.vertices[i]
            ni = self.vertices[adj[i].indices.reshape(-1, mm)]  # neighbors

            # Fit a quadratic function centered on the current vertex using its
            # tangent vectors (say, u and v) as basis. The "function values" are
            # the distances from each neighbor to its projection on the tangent
            # plane

            # Get coordinates in basis (u, v) by projecting each neighbor onto the
            # tangent plane
            nivi = ni - vi[:, None]
            uv = np.squeeze(vt[i, :, None] @ nivi[:, None].swapaxes(2, 3))
            # Quadratic features
            A = np.concatenate(
                (uv**2, 2 * np.prod(uv, axis=1, keepdims=True)), axis=1
            ).swapaxes(1, 2)
            # Function values
            b = np.squeeze(nivi @ vn[i][..., None])

            # Least squares solution
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            x = np.squeeze(
                Vt.swapaxes(1, 2) @ (U.swapaxes(1, 2) @ b[..., None] / S[..., None])
            )

            # Estimate the coefficients of the second fundamental form
            # Hessian
            H_uv[i, 0, 0] = 2 * x[:, 0]
            H_uv[i, 1, 1] = 2 * x[:, 1]
            H_uv[i, 0, 1] = H_uv[i, 1, 0] = 2 * x[:, 2]

            H_uv[i] = self._second_fundamental_form_coefficients(vi, ni, vt[i], vn[i])

            # # only needed for bad conditioning?
            # rsq = A[:,:2].sum(1) # u**2 + v**2
            # k = b/rsq
            # kmin[i] = k.min()
            # kmax[i] = k.max()

        # Estimate curvature from the second fundamental form
        # (see https://en.wikipedia.org/wiki/Principal_curvature)
        # D = principal curvatures
        # E = principal directions, i.e., the directions of maximum and minimum
        #     curvatures.
        # Positive curvature means that the surface bends towards the normal (e.g.,
        # in a sulcus)
        D, E = np.linalg.eigh(H_uv)
        # sort in *descending* order
        D = D[:, ::-1]
        E = E[:, ::-1]
        return D, E

    @staticmethod
    def _second_fundamental_form_coefficients(vi, ni, vit, vin):
        """

        vi : vertex at which to estimate curvature
        ni : neighbors
        vit : vertex tangent plane vectors
        vin : vector normal
        """
        vi = np.atleast_2d(vi)
        n_vi = vi.shape[0]
        ni = np.atleast_3d(ni)

        # Fit a quadratic function centered on the current vertex using its
        # tangent vectors (say, u and v) as basis. The "function values" are
        # the distances from each neighbor to its projection on the tangent
        # plane
        nivi = ni - vi[:, None]
        uv = np.squeeze(vit[..., None] @ nivi[:, None].swapaxes(2, 3))
        # Quadratic features
        A = np.concatenate(
            (uv**2, 2 * np.prod(uv, axis=1, keepdims=True)), axis=1
        ).swapaxes(1, 2)
        # Function values
        b = np.squeeze(nivi @ vin[..., None])

        # Least squares solution
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        x = np.squeeze(
            Vt.swapaxes(1, 2) @ (U.swapaxes(1, 2) @ b[..., None] / S[..., None])
        )

        # Estimate the coefficients of the second fundamental form
        # Hessian
        H_uv = np.zeros((n_vi, 2, 2))
        H_uv[0, 0] = 2 * x[:, 0]
        H_uv[1, 1] = 2 * x[:, 1]
        H_uv[0, 1] = H_uv[1, 0] = 2 * x[:, 2]

        return H_uv.squeeze()

    def compute_curvatures(self, smooth_iter: int = 10):
        """Compute principal curvatures. Optionally calculate mean and Gaussian
        curvatures as well.

        Parameters
        ----------
        niter : int:
            Number of smoothing iterations. Defaults to 10.

        Returns
        -------
        curvature : dict
            k1,k2 : principal curvatures, i.e., the directions of maximum and
                    minimum curvature, respectively.
            H     : mean curvature
            K     : Gaussian curvature
        """
        D, _ = self.compute_principal_curvatures()
        A = self.compute_vertex_adjacency(with_diag=True)
        k1, k2 = self.iterative_spatial_smoothing(D, smooth_iter, A).T
        H = 0.5 * (k1 + k2)
        K = k1 * k2
        # H = self.iterative_spatial_smoothing(A, D.mean(1), niter)
        # K = self.iterative_spatial_smoothing(A, D.prod(1), niter)
        self.curv = Curvature(k1=k1, k2=k2, H=H, K=K)

    # def smooth(self, data):

    def iterative_spatial_smoothing(
        self, data: npt.NDArray, niter: int, A=None, nn=None
    ):
        """Perform iterative spatial smoothing of `data` on the mesh defined by
        the adjacency matrix `A`.

        Parameters
        ----------
        data : npt.NDArray
            Data to smooth.
        niter : int
            Number of smoothing iterations.
        nn : None | npt.NDArray
            Number of neighbors of each node. If not specified, it is calculated
            from `A`. Defaults to None.

        Returns
        -------
        data : npt.NDArray
            The smoothed data.

        Notes
        -----
        This function mimics Freesurfer's `MRISaverageCurvatures`.
        """
        A = self.compute_vertex_adjacency(with_diag=True) if A is None else A
        if niter > 0:
            nn = nn if nn is not None else np.array(A.sum(1)).squeeze()
            nn = nn[:, None] if data.ndim > nn.ndim else nn
            return self.iterative_spatial_smoothing(A @ data / nn, niter - 1, A, nn)
        elif niter == 0:
            return data
        else:
            raise ValueError("`navgs` should be >= 0")

    def apply_affine(self, affine: npt.NDArray, move: bool = True) -> npt.NDArray:
        """Apply an affine to an array of points.

        Parameters
        ----------
        affine : npt.NDArray
            A 4x4 array defining the vox2world transformation.
        move : bool
            If True (default), apply translation.

        Returns
        -------
        out_coords : shape = (3,) | (n,
            Transformed point(s).
        """

        # apply rotation & scale
        out_coords = np.dot(self.vertices, affine[:3, :3].T)
        # apply translation
        if move:
            out_coords += affine[:3, 3]

        return out_coords

    def interpolate_to_nodes(
        self, vol: npt.NDArray, affine: npt.NDArray, order: int = 3
    ) -> npt.NDArray:
        """Interpolate values from a volume to surface node positions.

        Parameters
        ----------
        vol : npt.NDArray
            A volume array as read by e.g., nib.load(image).get_fdata()
        affine: npt.NDArray
            A 4x4 array storing the vox2world transformation of the image
        order: int
            Interpolation order (0-5)

        Returns
        -------
        values_at_coords: npt.NDArray
                        An Nx1 array of intensity values at each node

        """

        # Map node coordinates to volume
        inv_affine = np.linalg.inv(affine)
        # NOTE: I'm not sure if we should always work in scanner
        # RAS or not, but here I'm assuming the vertices in the
        # object are in surface RAS
        vox_coords = self.apply_affine(
            inv_affine, self.vertices + self.metadata["cras"]
        )

        # Deal with edges ala simnibs
        im_shape = vol.shape
        for i, s in enumerate(im_shape):
            vox_coords[(vox_coords[:, i] > -0.5) * (vox_coords[:, i] < 0), i] = 0.0
            vox_coords[(vox_coords[:, i] > s - 1) * (vox_coords[:, i] < s - 0.5), i] = (
                s - 1
            )

        # Keeping the map_coordinates options exposed in case we want to change these
        return map_coordinates(
            vol, vox_coords.T, order=order, mode="constant", cval=0.0, prefilter=True
        )

    @classmethod
    def from_freesurfer_subject_dir(cls, subject_dir, surface, read_metadata=True):
        v, f, metadata = nib.freesurfer.read_geometry(
            Path(subject_dir) / "surf" / surface, read_metadata=read_metadata
        )
        return Surface(v, f, metadata)
