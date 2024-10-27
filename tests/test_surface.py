import copy

import numpy as np
import pytest

from cortech.surface import Surface

import cortech.utils


@pytest.fixture
def diamond(diamond):
    return Surface(*diamond)

@pytest.fixture
def sphere_surface(sphere):
    return Surface(*sphere)


def sph_to_cart(theta, phi):
    """
    points : r, theta, phi in columns
    """
    theta = np.atleast_2d(theta)
    phi = np.atleast_2d(phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.squeeze(np.stack([x,y,z], axis=1))


class TestSurface:
    def test_create_surface(self, sphere):
        s = Surface(*sphere)
        np.testing.assert_allclose(s.vertices, sphere[0])
        np.testing.assert_allclose(s.faces, sphere[1])

    @pytest.mark.parametrize("include_self", [False, True])
    def test_compute_vertex_adjacency(self, include_self, diamond, diamond_adjacency_matrix):
        a = diamond.compute_vertex_adjacency(include_self)
        a_true = diamond_adjacency_matrix
        if include_self:
            a_true = a_true + np.eye(diamond.n_vertices)
        np.testing.assert_array_equal(a.todense(), a_true)

    def test_compute_face_barycenters(self, diamond, diamond_barycenters):
        b = diamond.compute_face_barycenters()
        np.testing.assert_allclose(b, diamond_barycenters)

    def test_compute_face_normals(self, diamond):
        n = diamond.compute_face_normals()
        n_true = cortech.utils.normalize(diamond.compute_face_barycenters(), axis=1)
        np.testing.assert_allclose(n, n_true)

    def test_compute_vertex_normals(self, diamond):
        n = diamond.compute_vertex_normals()
        n_true = cortech.utils.normalize(diamond.vertices, axis=1)
        np.testing.assert_allclose(n, n_true)

    def test_compute_principal_curvatures(self):
        pass

    # @pytest.mask.parametrize("radius", [0.5, 1.0, 5.0])
    # def test_compute_curvature(self, radius):
    #     sphere = Surface(*fibonacci_sphere(10000, radius))
    #     curv = sphere.compute_curvature()
    #     curvs = sphere.compute_curvature(smooth_iter=10)

    #     k1_true = -1.0/radius
    #     k2_true = k1_true
    #     H_true = k1_true
    #     K_true = 2 * H_true

    #     theta_resolution = 200
    #     phi_resolution = 100
    #     theta = np.linspace(0, 2 * np.pi, theta_resolution)
    #     phi = np.linspace(0, np.pi, phi_resolution)


    #     p = sph_to_cart(np.repeat(theta, len(phi)), np.tile(phi, len(theta)))

    #     p = 0.5 * p
    #     v,f = convex_hull(p)


    #     pd = pv.make_tri_mesh(v, f)
    #     pd.save("test3.vtk")

    #     sphere = pv.Sphere(0.5, theta_resolution=200, phi_resolution=100)
    #     surf = Surface(sphere.points, sphere.faces.reshape(-1, 4)[:,1:])
    #     curv = surf.compute_curvature()

    #     np.testing.assert_allclose(curv.k1, k1_true)
    #     np.testing.assert_allclose(curv.k2, k2_true, atol=0.4)
    #     np.testing.assert_allclose(curv.H, H_true, atol=0.4)
    #     np.testing.assert_allclose(curv.K, K_true, atol=0.8)

    #     np.testing.assert_allclose(curvs.k1, k1_true)
    #     np.testing.assert_allclose(curvs.k2, k2_true)
    #     np.testing.assert_allclose(curvs.H, H_true)
    #     np.testing.assert_allclose(curvs.K, K_true)


    # def test_convex_hull(self, diamond):
    #     p = np.concatenate(
    #         (diamond.vertices, np.array([[0.0,0.0,0.0]]), np.array([[1.0,1.0,1.0]])), axis=0)
    #     diamond_copy = diamond.

    #     hull = Surface.convex_hull(p)


    # def test_k_ring_neighbors():


    #     s = Surface(vertices, faces)
    #     knn,kr = s.k_ring_neighbors(1, 0)
    #     knn[0] == 0
    #     knn[1:5] ==
    #     n = s.k_ring_neighbors(2, 0)


    def test_remove_self_intersections(self):
        pass

    def test_self_intersections(self):
        pass


    def test_connected_components(self):
        pass



    def test_points_inside_surface(self, diamond, diamond_barycenters, eps=1e-6):
        # Move outwards
        is_inside = diamond.points_inside_surface(diamond_barycenters * (1+eps))
        np.testing.assert_allclose(is_inside, False)
        # Move inwards
        is_inside = diamond.points_inside_surface(diamond_barycenters * (1-eps))
        np.testing.assert_allclose(is_inside, True)



    def test_shape_smooth(self):
        pass

    def test_taubin_smooth(self):
        pass

    def test_gaussian_smooth(self):
        pass

    def test_get_triangle_neighbors(self):
        pass

    def test_get_nearest_triangles_on_surface(self):
        pass

    def test_project_points_to_surface(self):
        pass

    def test_prune(self, diamond):
        # *Prepend* fake vertices, adjust faces accordingly, and prune to
        # recover the original mesh.
        d = copy.deepcopy(diamond)
        d.vertices = np.concatenate((np.ones_like(d.vertices), d.vertices), axis=0)
        d.faces += diamond.n_vertices
        d.prune()

        np.testing.assert_allclose(d.vertices, diamond.vertices)
        np.testing.assert_allclose(d.faces, diamond.faces)

    def test_from_freesurfer_subject_dir(self):
        pass