import time

import numpy as np
from scipy.spatial import cKDTree

import simnibs
from simnibs.mesh_tools import cython_msh



# hemisphere object (white, pial surfaces)
# -> correct self-intersections in each surface
# -> decouple white and pial surfaces
# -> produce volume mesh
# ->


from simnibs.mesh_tools.meshing import _mesh_surfaces
from simnibs.mesh_tools.mesh_io import make_surface_mesh



def volume_mesh(self):

    optimize = False

    surfaces = [
        make_surface_mesh(self.white.vertices, self.white.faces + 1),
        make_surface_mesh(self.pial.vertices, self.pial.faces + 1),
    ]

    return _mesh_surfaces(
        surfaces,
        subdomains = [(0,1), (1,2), (2,0)],
        facet_angle = 30,
        facet_size = 10,
        facet_distance = 0.1,
        cell_radius_edge_ratio = 2,
        cell_size = 10,
        optimize = optimize,
    )




def prepare_for_field_line_tracing(mesh, potential_limits=None):
    """Solve a PDE where a certain potential is set on the white and pial
    surfaces.
    """

    wm_vertices = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1001, :3])-1
    gm_vertices = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3])-1

    cond = simnibs.simulation.opt_struct.SimuList(mesh).cond2elmdata()

    potential_limits = potential_limits or dict(white=1000, pial=0)

    # Define boundary conditions
    dirichlet = simnibs.simulation.fem.DirichletBC(
        np.concatenate((wm_vertices+1, gm_vertices+1)),
        np.concatenate((
            np.full(wm_vertices.size, potential_limits["white"]),
            np.full(gm_vertices.size, potential_limits["pial"])))
    )

    # Solve
    laplace_eq = simnibs.simulation.fem.FEMSystem(mesh, cond, dirichlet, store_G=True)
    potential = laplace_eq.solve()
    potential = np.clip(potential, potential_limits["pial"], potential_limits["white"])

    potential_elm = potential[mesh.elm.node_number_list-1]

    # Compute E field
    E_elm = - np.sum(laplace_eq._G * potential_elm[..., None], 1)
    E_mag_elm = np.linalg.norm(E_elm, axis=1)

    # Interpolate E field to nodes

    # SPR interpolation matrix
    M = mesh.interp_matrix(
        mesh.nodes.node_coord, out_fill='nearest', th_indices=None, element_wise=True
    )
    E = M @ E_elm
    E_mag = np.linalg.norm(E, axis=1)


    # Normalized field vector
    N = np.divide(E, E_mag[:, None], where=E_mag[:, None]>0)
    N_elm = N[mesh.elm.node_number_list-1]


    is_valid = E_mag.squeeze() > 1e-8
    # E = E[is_valid]
    # E_mag = E_mag[is_valid]

    print("E magnitude (minimum)", E_mag.min())


    mesh.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(potential, "potential (node)", mesh))
    mesh.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(E, "E (node)", mesh))
    mesh.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(E_mag, "|E| (node)", mesh))
    mesh.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(N, "N (node)", mesh))
    mesh.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(is_valid, "valid (node)", mesh))

    mesh.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(potential_elm, "potential (elm)", mesh))
    mesh.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(E_elm, "E (elm)", mesh))
    mesh.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(E_mag_elm, "|E| (elm)", mesh))
    mesh.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(N_elm, "N (elm)", mesh))



    return


def prepare_for_tetrahedron_with_points(mesh):
    indices_tetra = mesh.elm.tetrahedra
    nodes_tetra = np.array(mesh.nodes[mesh.elm[indices_tetra]], float)
    th_baricenters = nodes_tetra.mean(1)

    # Calculate a few things we will use later
    _, faces_tetra, adjacency_list = mesh.elm.get_faces(indices_tetra)
    faces_tetra = np.array(faces_tetra, dtype=int)
    adjacency_list = np.array(adjacency_list, dtype=int)

    kdtree = cKDTree(th_baricenters)

    return faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra


def tetrahedron_with_points(points, faces_tetra, nodes_tetra, adjacency_list, indices_tetra, init_tetra):



    tetra_index = cython_msh.find_tetrahedron_with_points(
        np.array(points, float), nodes_tetra, init_tetra, faces_tetra, adjacency_list
    )


    # calculate baricentric coordinates
    inside = tetra_index != -1

    M = np.transpose(
        nodes_tetra[tetra_index[inside], :3] - nodes_tetra[tetra_index[inside], 3, None],
        (0, 2, 1)
    )
    baricentric = np.zeros((len(points), 4), dtype=float)
    baricentric[inside, :3] = np.linalg.solve(
        M, points[inside] - nodes_tetra[tetra_index[inside], 3]
    )
    baricentric[inside, 3] = 1 - np.sum(baricentric[inside], axis=1)

    # Return indices
    tetra_index[inside] = indices_tetra[tetra_index[inside]]

    return tetra_index-1, baricentric



def euler_forward(mesh_tets, gm_vertices, is_valid, potential, potential_limits, h_max=0.1, thickness=None):

    h_max = 0.1 # maximum stepsize (in mm)

    # collect necessary quantities
    # nodes
    N = mesh.field["N (node)"]
    E_mag = mesh.field["|E|"]
    # elements
    pot_elm = mesh.field["potential (elm)"]
    N_elm = mesh.field("N (elm)")
    E_mag_elm = mesh.field["|E| (elm)"]

    t0 = time.perf_counter()

    # intialize the random walk to tetrahedron with closest baricenter
    # subsequent iterations use the previously found tetrahedron at starting
    # point
    faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra = prepare_for_tetrahedron_with_points(mesh_tets)
    t1 = time.perf_counter()
    print("time elapsed (initialize tetrahedron search)", t1-t0, "seconds")


    valid_gm = gm_vertices[is_valid[gm_vertices]]

    # bbox = np.array([[-66, -27, 14], [-22, -7, 54]])

    # x = mesh_tets.nodes.node_coord[valid_gm]
    # in_bbox = np.all((x >= bbox[0]) & (x <= bbox[1]), 1)

    # valid_gm = valid_gm[in_bbox]


    thickness = np.zeros(valid_gm.size)

    # iteration 0
    y = potential[valid_gm]
    pos = mesh_tets.nodes.node_coord[valid_gm]

    # target gradient in V: we aim at stepping 1 % of the way at each iteration
    V_stepsize = 0.01 * np.abs(np.diff(list(potential_limits.values())))[0]

    # Starting position for walking algorithm: the closest baricenter
    _, tetra_index = np.array(kdtree.query(pos), int)

    # target_frac = 0.5


    sampled_positions = [pos.copy()]


    # Initialize
    still_valid = tetra_index>0
    tmp = tetra_index>0

    dydt = N[valid_gm[still_valid]]
    h = np.minimum(h_max, V_stepsize / E_mag[valid_gm[still_valid]])

    valid_iterations = np.zeros(valid_gm.size, int)

    thickness_increment = [np.zeros(valid_gm.size)]

    i = 0
    while True:
        i += 1

        # Forward Euler step
        pos_next = pos[still_valid] - h[:, None] * dydt

        # idx = thickness[still_valid]>target_frac
        # y_prev[still_valid][idx] + y[still_valid][idx]

        # Find tetrahedron in which each point is located
        # index is zero-based!
        tetra_index, coo_bari = tetrahedron_with_points(
            pos_next, faces_tetra, nodes_tetra, adjacency_list, indices_tetra, tetra_index[tmp]
        )
        # assert np.all(tetra_index >= 0)
        tmp = tetra_index>0

        if not tmp.any():
            print(f"no more valid vertices. exiting at {i}")
            break

        # Accept move and update thickness for points which are still inside
        # the domain
        still_valid[still_valid] = tmp
        pos[still_valid] = pos_next[tmp]

        x = np.zeros(valid_gm.size)
        x[still_valid] = h[tmp]
        thickness_increment.append(x)

        thickness[still_valid] += h[tmp]

        valid_iterations[still_valid] += 1


        # we could calculate the exact point where the field line crosses the
        # mesh but perhaps that is not really worth it given that the lines
        # seem to terminate at >97% thickness

        # y_tmp = np.sum(potential_tetrahedra[tetra_index[tmp]] * coo_bari[tmp], 1)
        # idx = y_tmp > target_frac
        # y[still_valid][tmp]

        # linear interpolation

        # REMOVE; only for diagnostics...
        y[still_valid] = np.sum(pot_elm[tetra_index[tmp]] * coo_bari[tmp], 1)
        print(f"{i:3d} : {y[still_valid].min():10.3f} {y[still_valid].mean():10.3f} {y[still_valid].max():10.3f} {still_valid.sum()/len(still_valid):10:3f}")

        dydt = np.sum(N_elm[tetra_index[tmp]] * coo_bari[tmp, :, None], 1)
        dydt_norm = np.linalg.norm(dydt, axis=1, keepdims=True)
        dydt = np.divide(dydt, dydt_norm, where=dydt_norm>0) # check zeros...

        # update h for next iteration: sample |E| at current position to
        # determine step size
        E_mag_sample = np.sum(E_mag_elm[tetra_index[tmp]] * coo_bari[tmp], 1)
        h = np.minimum(h_max, V_stepsize / E_mag_sample)

        sampled_positions.append(pos.copy())

    thickness_increment = np.array(thickness_increment)
    all_pos = np.array(sampled_positions)
    # all_y = np.array(all_y)

    print("time elapsed (trace field lines)", time.perf_counter()-t1, "seconds")

    return all_pos, thickness_increment


# fit parametric for (polynomial of degree n); n = 4 seems to work okay

def parameterize_field_lines(all_pos, h_history, valid_iterations, order=5):

    # parameteric curve is valid for [0, ..., thickness] at each point

    n_vertices = len(valid_iterations)
    min_number_of_points = order ** 3

    cumulative_thickness = h_history.cumsum(0)

    parameters = np.zeros((n_vertices, order, 3))
    residuals = np.zeros((n_vertices, 3))

    for i in range(n_vertices):
        if valid_iterations[i] < min_number_of_points:
            continue

        c = all_pos[0, i] # intercept
        Y = all_pos[1:valid_iterations[i]+1, i]
        t = cumulative_thickness[1:valid_iterations[i]+1, i] # coordinates

        A = np.stack([t**i for i in range(1,order+1)], axis=1)
        b = Y-c
        X, res, _, _ = np.linalg.lstsq(A, b)
        parameters[i] = X
        residuals[i] = res

    return parameters, residuals