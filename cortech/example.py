import pyvista as pv

from cortech.surface import Surface
from cortech.cortex import Hemisphere

# setup lh of a subject
sub04_lh = Hemisphere.from_freesurfer_subject_dir(
    "/home/jesperdn/INN_JESPER/nobackup/projects/anateeg/freesurfer/sub-04",
    "lh",
)

bert = Hemisphere.from_freesurfer_subject_dir(
    "/mnt/depot64/freesurfer/freesurfer.7.4.0/subjects/bert",
    "lh",
)

# Visualization

# visualize with kwargs
curv = bert.white.compute_curvature()
bert.white.plot(
    curv.H,
    mesh_kwargs=dict(show_edges=True),
    plotter_kwargs=dict(notebook=False)
)


# Smoothing

v = bert.white.gaussian_smooth(n_iter=25)
gs = Surface(v, bert.white.faces)
gs.plot()

v = bert.white.taubin_smooth(n_iter=25)
ts = Surface(v, bert.white.faces)
ts.plot()



# ---

sub04_lh.white.remove_self_intersections()
sub04_lh.pial.remove_self_intersections()

sub04_lh.decouple_brain_surfaces()



wm_curv = sub04_lh.white.compute_curvature()
white_H = sub04_lh.white.iterative_spatial_smoothing(wm_curv.H, 10)

pial_curv = sub04_lh.pial.compute_curvature()
pial_H = sub04_lh.pial.iterative_spatial_smoothing(pial_curv.H, 10)

# compute the average curvature (white/2 + pial/2)
curv = sub04_lh.compute_average_curvature(curv_kwargs=dict(smooth_iter=10))

# setup lh of a fsaverage
# 'fsaverage' is special; it just grabs from $FS_HOME/subjects/fsaverage
fsavg_lh = Hemisphere.from_freesurfer_subject_dir("fsaverage", "lh")

# compute projection from subject to fsaverage. This is stored internally
sub04_lh.spherical_registration.compute_projection(fsavg_lh.spherical_registration)
# apply projection
resampled_white_H = sub04_lh.spherical_registration.resample(white_H)
resampled_pial_H = sub04_lh.spherical_registration.resample(pial_H)

# Visualize

# show on subject

m = pv.make_tri_mesh(sub04_lh.white.vertices, sub04_lh.white.faces)
m["H"] = white_H
m.plot(show_edges=False)

m = pv.make_tri_mesh(sub04_lh.pial.vertices, sub04_lh.pial.faces)
m["H"] = pial_H
m.plot(show_edges=False)

# show on fsaverage

q = pv.make_tri_mesh(fsavg_lh.white.vertices, fsavg_lh.white.faces)
q["H"] = resampled_white_H
q.plot()

q = pv.make_tri_mesh(fsavg_lh.pial.vertices, fsavg_lh.pial.faces)
q["H"] = resampled_pial_H
q.plot()

# show on sphere.reg

q = pv.make_tri_mesh(
    fsavg_lh.spherical_registration.vertices,
    fsavg_lh.spherical_registration.faces
)
q["H"] = resampled_white_H
q.plot()

q = pv.make_tri_mesh(
    fsavg_lh.spherical_registration.vertices,
    fsavg_lh.spherical_registration.faces
)
q["H"] = resampled_pial_H
q.plot()
