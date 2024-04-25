import itertools
import os
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
        self, vertices: npt.NDArray, faces: npt.NDArray, metadata=None
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.metadata = metadata

    def is_valid(self):
        # and check that n_faces and n_vertices match for whatever number of vertices per face....
        # only valid for triangulated surfaces
        return self.n_faces == self.n_vertices * 2 - 4

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        value = np.atleast_2d(value)
        assert value.ndim == 2
        self._faces = value
        self.n_faces, self.vertices_per_face = value.shape

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = np.atleast_2d(value)
        assert value.ndim == 2
        self._vertices = value
        self.n_vertices, self.n_dim = value.shape

    def as_mesh(self):
        return self.vertices[self.faces]

    def compute_vertex_adjacency(self, with_diag=False):
        """Make sparse adjacency matrix for vertices with connections `tris`."""
        pairs = list(itertools.combinations(np.arange(self.faces.shape[1]), 2))
        row_ind = np.concatenate([self.faces[:, i] for p in pairs for i in p])
        col_ind = np.concatenate([self.faces[:, i] for p in pairs for i in p[::-1]])

        data = np.ones_like(row_ind)
        A = scipy.sparse.csr_array(
            (data / 2, (row_ind, col_ind)), shape=(self.n_vertices, self.n_vertices)
        )

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
        n = self.n_vertices
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
        uv = np.squeeze(vit[:, :, None] @ nivi[:, None].swapaxes(2, 3))
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
        H_uv[:, 0, 0] = 2 * x[:, 0]
        H_uv[:, 1, 1] = 2 * x[:, 1]
        H_uv[:, 0, 1] = H_uv[:, 1, 0] = 2 * x[:, 2]

        return H_uv.squeeze()

    def compute_curvature(
        self, percentile_clip_range=(0.1, 99.9), smooth_iter: int = 0
    ):
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

        if percentile_clip_range is not None:
            clip_range = np.percentile(D, percentile_clip_range, axis=0)
            for i, (low, hi) in enumerate(clip_range.T):
                D[:, i] = np.clip(D[:, i], low, hi)

        k1, k2 = D.T
        H = D.mean(1)
        K = D.prod(1)

        if smooth_iter > 0:
            A = self.compute_vertex_adjacency(with_diag=True)

            k1 = self.iterative_spatial_smoothing(k1, smooth_iter, A)
            k2 = self.iterative_spatial_smoothing(k2, smooth_iter, A)
            H = self.iterative_spatial_smoothing(D.mean(1), smooth_iter, A)
            K = self.iterative_spatial_smoothing(D.prod(1), smooth_iter, A)

        return Curvature(k1=k1, k2=k2, H=H, K=K)

        # store the curvature directions as well
        # self.curv_vec = Curvature(k1=E[:, 0], k2=[E[:, 1]])

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

    @staticmethod
    def apply_affine(
        vertices: npt.NDArray, affine: npt.NDArray, move: bool = True
    ) -> npt.NDArray:
        """Apply an affine to an array of points.

        Parameters
        ----------
        vertices : npt.NDArray
            Node coordinates
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
        out_coords = np.dot(vertices, affine[:3, :3].T)
        # apply translation
        if move:
            out_coords += affine[:3, 3]

        return out_coords

    def interpolate_to_nodes(
        self,
        vol: npt.NDArray,
        affine: npt.NDArray,
        order: int = 3,
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

        # Check if metadata exists and if cras exists
        if self.metadata is not None and "cras" in self.metadata:
            vertices = self.vertices + self.metadata["cras"]
        else:
            vertices = self.vertices

        # Map node coordinates to volume
        inv_affine = np.linalg.inv(affine)
        vox_coords = self.apply_affine(vertices, inv_affine)

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

    def get_triangle_neighbors(self):
        """For each point get its neighboring triangles (i.e., the triangles to
        which it belongs).

        PARAMETERS
        ----------
        tris : ndarray
            Array describing a triangulation with size (n, 3) where n is the number
            of triangles.
        nr : int
            Number of points. If None, it is inferred from `tris` as tris.max()+1
            (default = None).

        RETURNS
        -------
        pttris : ndarray
            Array of arrays where pttris[i] are the neighboring triangles of the
            ith point.
        """
        rows = self.faces.ravel()
        cols = np.repeat(np.arange(self.n_faces), self.vertices_per_face)
        data = np.ones_like(rows)
        csr = scipy.sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.n_vertices, self.n_faces)
        ).tocsr()
        return np.array(np.split(csr.indices, csr.indptr[1:-1]), dtype=object)

    def get_nearest_triangles_on_surface(
        self, points: npt.NDArray, n: int = 1, subset=None, return_index: bool = False
    ):
        """For each point in `points` get the `n` nearest nodes on `surf` and
        return the triangles to which these nodes belong.

        points : ndarray
            Points for which we want to find the candidate triangles. Shape (n, d)
            where n is the number of points and d is the dimension.
        surf : dict
            Dictionary with keys points and tris corresponding to the nodes and
            triangulation of the surface, respectively.
        n : int
            Number of nearest vertices in `self` to consider for each point in
            `points`.
        subset : array-like
            Use only a subset of the vertices in `surf`. Should be indices *not* a
            boolean mask!
        return_index : bool
            Return the index (or indices if n > 1) of the nearest vertex in `surf`
            for each point in `points`.

        RETURNS
        -------
        pttris : list
            Point to triangle mapping.
        """
        assert isinstance(n, int) and n >= 1

        surf_points = self.vertices if subset is None else self.vertices[subset]
        tree = scipy.spatial.cKDTree(surf_points)
        _, ix = tree.query(points, n)
        if subset is not None:
            ix = subset[ix]  # ensure ix indexes into surf['points']
        pttris = self.get_triangle_neighbors()[ix]
        if n > 1:
            pttris = list(map(lambda x: np.unique(np.concatenate(x)), pttris))
        return (pttris, ix) if return_index else pttris

    def project_points_to_surface(
        self,
        points: npt.NDArray,
        pttris: list | np.ndarray,
        return_all_projections: bool = False,
    ):
        """Project each point in `points` to the closest point on the surface
        described by `surf` restricted to the triangles in `pttris`.

        PARAMETERS
        ----------
        points : ndarray
            Array with shape (n, d) where n is the number of points and d is the
            dimension.
        pttris : ndarray | list
            If a ragged/nested array, the ith entry contains the triangles against
            which the ith point will be tested.
        return_all_projections : bool
            Whether to return all projection results (i.e., the projection of a
            point on each of the triangles which it was tested against) or only the
            projection on the closest triangle.

        RETURNS
        -------
        tris : ndarray
            The index of the triangle onto which a point was projected.
        weights : ndarray
            The linear interpolation weights resulting in the projection of a point
            onto a particular triangle.
        projs :
            The coordinates of the projection of a point on a triangle.
        dists :
            The distance of a point to its projection on a triangle.

        NOTES
        -----
        The cost function to be minimized is the squared distance between a point
        P and a triangle T

            Q(s,t) = |P - T(s,t)|**2 =
                = a*s**2 + 2*b*s*t + c*t**2 + 2*d*s + 2*e*t + f

        The gradient

            Q'(s,t) = 2(a*s + b*t + d, b*s + c*t + e)

        is set equal to (0,0) to find (s,t).

        REFERENCES
        ----------
        https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

        """
        npttris = list(map(len, pttris))
        pttris = np.concatenate(pttris)

        m = self.as_mesh()
        v0 = m[:, 0]  # Origin of the triangle
        e0 = m[:, 1] - v0  # s coordinate axis
        e1 = m[:, 2] - v0  # t coordinate axis

        # Vector from point to triangle origin (if reverse, the negative
        # determinant must be used)
        rep_points = np.repeat(points, npttris, axis=0)
        w = v0[pttris] - rep_points

        a = np.sum(e0**2, 1)[pttris]
        b = np.sum(e0 * e1, 1)[pttris]
        c = np.sum(e1**2, 1)[pttris]
        d = np.sum(e0[pttris] * w, 1)
        e = np.sum(e1[pttris] * w, 1)
        # f = np.sum(w**2, 1)

        # s,t are so far unnormalized!
        s = b * e - c * d
        t = b * d - a * e
        det = a * c - b**2

        # Project points (s,t) to the closest points on the triangle (s',t')
        sp, tp = np.zeros_like(s), np.zeros_like(t)

        # We do not need to check a point against all edges/interior of a triangle.
        #
        #          t
        #     \ R2|
        #      \  |
        #       \ |
        #        \|
        #         \
        #         |\
        #         | \
        #     R3  |  \  R1
        #         |R0 \
        #    _____|____\______ s
        #         |     \
        #     R4  | R5   \  R6
        #
        # The code below is equivalent to the following if/else structure
        #
        # if s + t <= 1:
        #     if s < 0:
        #         if t < 0:
        #             region 4
        #         else:
        #             region 3
        #     elif t < 0:
        #         region 5
        #     else:
        #         region 0
        # else:
        #     if s < 0:
        #         region 2
        #     elif t < 0
        #         region 6
        #     else:
        #         region 1

        # Conditions
        st_l1 = s + t <= det
        s_l0 = s < 0
        t_l0 = t < 0

        # Region 0 (inside triangle)
        i = np.flatnonzero(st_l1 & ~s_l0 & ~t_l0)
        deti = det[i]
        sp[i] = s[i] / deti
        tp[i] = t[i] / deti

        # Region 1
        # The idea is to substitute the constraints on s and t into F(s,t) and
        # solve, e.g., here we are in region 1 and have Q(s,t) = Q(s,1-s) = F(s)
        # since in this case, for a point to be on the triangle, s+t must be 1
        # meaning that t = 1-s.
        i = np.flatnonzero(~st_l1 & ~s_l0 & ~t_l0)
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        numer = cc + ee - (bb + dd)
        denom = aa - 2 * bb + cc
        sp[i] = np.clip(numer / denom, 0, 1)
        tp[i] = 1 - sp[i]

        # Region 2
        i = np.flatnonzero(~st_l1 & s_l0)  # ~t_l0
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        tmp0 = bb + dd
        tmp1 = cc + ee
        j = tmp1 > tmp0
        j_ = ~j
        k, k_ = i[j], i[j_]
        numer = tmp1[j] - tmp0[j]
        denom = aa[j] - 2 * bb[j] + cc[j]
        sp[k] = np.clip(numer / denom, 0, 1)
        tp[k] = 1 - sp[k]
        sp[k_] = 0
        tp[k_] = np.clip(-ee[j_] / cc[j_], 0, 1)

        # Region 3
        i = np.flatnonzero(st_l1 & s_l0 & ~t_l0)
        cc, ee = c[i], e[i]
        sp[i] = 0
        tp[i] = np.clip(-ee / cc, 0, 1)

        # Region 4
        i = np.flatnonzero(st_l1 & s_l0 & t_l0)
        aa, cc, dd, ee = a[i], c[i], d[i], e[i]
        j = dd < 0
        j_ = ~j
        k, k_ = i[j], i[j_]
        sp[k] = np.clip(-dd[j] / aa[j], 0, 1)
        tp[k] = 0
        sp[k_] = 0
        tp[k_] = np.clip(-ee[j_] / cc[j_], 0, 1)

        # Region 5
        i = np.flatnonzero(st_l1 & ~s_l0 & t_l0)
        aa, dd = a[i], d[i]
        tp[i] = 0
        sp[i] = np.clip(-dd / aa, 0, 1)

        # Region 6
        i = np.flatnonzero(~st_l1 & t_l0)  # ~s_l0
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        tmp0 = bb + ee
        tmp1 = aa + dd
        j = tmp1 > tmp0
        j_ = ~j
        k, k_ = i[j], i[j_]
        numer = tmp1[j] - tmp0[j]
        denom = aa[j] - 2 * bb[j] + cc[j]
        tp[k] = np.clip(numer / denom, 0, 1)
        sp[k] = 1 - tp[k]
        tp[k_] = 0
        sp[k_] = np.clip(-dd[j_] / aa[j_], 0, 1)

        # Distance from original point to its projection on the triangle
        projs = v0[pttris] + sp[:, None] * e0[pttris] + tp[:, None] * e1[pttris]
        dists = np.linalg.norm(rep_points - projs, axis=1)
        weights = np.column_stack((1 - sp - tp, sp, tp))

        if return_all_projections:
            tris = pttris
        else:
            # Find the closest projection
            indptr = [0] + np.cumsum(npttris).tolist()
            i = cortech.utils.sliced_argmin(dists, indptr)
            tris = pttris[i]
            weights = weights[i]
            projs = projs[i]
            dists = dists[i]

        return tris, weights, projs, dists

    @classmethod
    def from_freesurfer_subject_dir(cls, subject_dir, surface, read_metadata=True):
        if subject_dir == "fsaverage":
            fs_home = Path(os.environ["FREESURFER_HOME"])
            subject_dir = fs_home / "subjects" / subject_dir

        surf_file = Path(subject_dir) / "surf" / surface
        # FS is changing to gii, but slowly
        if not surf_file.exists():
            surf_file = surf_file.parent / (str(surf_file.name) + ".gii")
            surf_gii = nib.load(surf_file)
            v, f = surf_gii.agg_data()

            v = v.astype(float)
            # Getting information out of the gifti is a pain
            # I'll only get the cras
            cras_array = np.array(
                [
                    float(surf_gii.darrays[0].meta["VolGeomC_R"]),
                    float(surf_gii.darrays[0].meta["VolGeomC_A"]),
                    float(surf_gii.darrays[0].meta["VolGeomC_S"]),
                ]
            )
            metadata = {"cras": cras_array}
        else:
            v, f, metadata = nib.freesurfer.read_geometry(
                surf_file, read_metadata=read_metadata
            )
        return cls(v, f, metadata)


class SphericalRegistration(Surface):
    def __init__(
        self,
        vertices: npt.NDArray,
        faces: npt.NDArray,
        metadata=None,
    ) -> None:
        super().__init__(vertices, faces, metadata)
        # Ensure on unit sphere
        self.vertices = cortech.utils.normalize(self.vertices, axis=-1)

    def compute_projection(
        self,
        other: "SphericalRegistration",
        method: str = "linear",
        n_nearest_vertices: int = 3,
    ):
        """Project the vertices of the (registered) spherical represention of
        self onto other. That is, compute a mapping from surfaces in self to
        other.

        Create a morph map which allows morphing of values from the nodes in
        `surf_from` to `surf_to` by nearest neighbor or linear interpolation.

        A morph map is a sparse matrix with
        dimensions (n_points_surf_to, n_points_surf_from) where each row has
        exactly three entries that sum to one. It is created by projecting each
        point in `surf_to` onto closest triangle in `surf_from` and determining
        the barycentric coordinates.

        Testing all points against all triangles is expensive and inefficient,
        thus we compute an approximation by finding, for each point in
        `surf_to`, the `self.n` nearest nodes on `surf_from` and the triangles
        to which these points belong. We then test only against these triangles.

                PARAMETERS
        ----------
        self :
            The source mesh (i.e., the mesh to interpolate *from*).
        other :
            The target mesh (i.e., the mesh to interpolate *to*).

        """
        assert method in {"nearest", "linear"}

        match method:
            case "nearest":
                kdtree = scipy.spatial.cKDTree(self.vertices)
                cols = kdtree.query(other.vertices)[1]
                rows = np.arange(other.n_vertices)
                weights = np.ones(other.n_vertices)

            case "linear":
                # Find the triangle (on `self`) to which each vertex in `other`
                # projects and get the associated weights
                points_to_faces = self.get_nearest_triangles_on_surface(
                    other.vertices, n_nearest_vertices
                )
                tris, weights, _, _ = self.project_points_to_surface(
                    other.vertices,
                    points_to_faces,
                )
                rows = np.repeat(np.arange(other.n_vertices), other.n_dim)
                cols = self.faces[tris].ravel()
                weights = weights.ravel()

        self._mapping_matrix = scipy.sparse.csr_matrix(
            (weights, (rows, cols)), shape=(other.n_vertices, self.n_vertices)
        )

    def resample(self, values: npt.NDArray):
        if not hasattr(self, "_mapping_matrix"):
            raise RuntimeError(
                "Please compute the mapping matrix (using the `fit_to` method) first."
            )
        return self._mapping_matrix @ values
