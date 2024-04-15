import numpy as np
import numpy.typing as npt

import cortech.utils
from cortech.surface import Surface

from cortech.constants import Curvature

class Hemisphere:
    """A class containing surfaces delineating the white-gray matter boundary
    as well as the gray matter-CSF boundary.

    Additionally, it may contain information about layers...
    """
    def __init__(
        self,
        white: Surface,
        pial: Surface,
        # thickness: None | npt.NDArray = None,
        # curvature: None | npt.NDArray = None,
    ) -> None:

        self.white = white
        self.pial = pial

        self.compute_thickness()
        self.compute_average_curvature()

    @classmethod
    def from_freesurfer_subject_dir(
        cls, sub_dir, hemi, white="white", pial="pial", thickness="thickness", curv="avg_curv"
    ):
        assert hemi in {"lh", "rh"}

        white = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{white}")
        pial = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{pial}")

        # if thickness:
        #     thickness = nib.freesurfer.read_morph_data(surf_dir / f"{hemi}.{thickness}")
        # if curv:
        #     curv = {h: 0.5 * (white.curv[k] + pial.curv[k]) for k in white.curv}

        #     # white = nib.freesurfer.read_morph_data(surf_dir / "{hemi}.{curv}")
        #     # pial = nib.freesurfer.read_morph_data(surf_dir / "{hemi}.{curv}.pial")

        return cls(white, pial)

    @classmethod
    def from_simnibs_subject_dir(cls, sub_dir, hemi):
        assert hemi in {"lh", "rh"}

        raise NotImplementedError


    def compute_thickness(self) -> None:
        """Calculate thickness at each vertex of node-matched surfaces."""

        # FIXME this should be a better estimate than simple node-to-node
        # distance
        vi = self.white.vertices
        vo = self.pial.vertices
        return np.linalg.norm(vo-vi, axis=1)

    def compute_average_curvature(self):
        """Average curvature estimates of white and pial surfaces."""
        return Curvature(**{k: 0.5 * (getattr(self.white, k) + getattr(self.pial, k)) for k in self.white.curv._fields})


    def compute_equivolume_fraction(
        self,
        vol_frac: float = 0.5,
        curv: None | npt.NDArray = None,
        thickness: None | npt.NDArray = None,
    ):
        """Compute the distance fraction (between inner and outer surface whose
        distance is `thickness`) which yields the desired volume fraction at each
        position.

        Parameters
        ----------
        curv : npt.NDArray
            Curvature estimate at each position.
        thickness : npt.NDArray
            Thickness estimate at each position.
        vol_frac : float
            The desired volume fraction. Defaults to 0.5.

        Returns
        -------
        dist_frac : npt.NDArray
            Position-wise distance fraction which yields the desired volume
            fraction.

        Notes
        -----
        We expect positive curvature in sulci and negative curvature on gyral
        crowns.

        On gyral crowns, the curvature is negative because the surface bends away
        from the normals. Since a radius must be positive, we ignore the sign when
        computing the radius of the sphere which gives the desired volume fraction.

        In the sulci, the curvature is positive as the surface bends towards the
        normal. However, we are still placing the center of the sphere on the side
        of the white matter (there is nothing to distinguish concave and convex
        when we ignore the sign of the curvature). Assuming the sphere was placed
        on the pial side, we could find the desired volume fraction by first
        estimating the radius of the sphere at 1-frac (since 1-vol_frac from the
        outer side is vol_frac from the inner side) and then subtract this
        (relative to the inner radius) from the thickness to get a distance from
        the inner surface rather than the outer.

        References
        ----------
        Michiel Kleinnijenhuis et al. (2015). Diffusion tensor characteristics of
            gyrencephaly using high resolution diffusion MRI in vivo at 7T.
        """
        curv = self.compute_average_curvature() if curv is None else curv
        thickness = self.compute_thickness() if thickness is None else thickness

        # Name variables in accordance with the reference
        R = 1/np.abs(curv)
        R3 = R**3
        T = thickness
        r = np.zeros_like(curv)

        pos = curv > 0
        neg = curv < 0

        r[neg] = cortech.utils.compute_sphere_radius(vol_frac, T[neg], R[neg], R3[neg])
        r[pos] = cortech.utils.compute_sphere_radius(1-vol_frac, T[pos], R[pos], R3[pos])
        # r[curv == 0] should be `vol_frac` but we take care of this in `dist_frac` below

        # r is a radius in between R and R+T so subtract R to get back to distances
        # that relate to the thickness
        r -= R

        # Correct positive curvature positions
        r[pos] *= -1
        r[pos] += T[pos]

        # Ensure r is valid
        _ = np.clip(r, 0, T, out=r)

        # Now compute the pointwise distance fraction
        dist_frac = np.full_like(r, vol_frac)
        _ = np.divide(r, T, out=dist_frac, where=T>0)

        return dist_frac


    def _place_layer_at_distance_fraction(self, frac: float | npt.NDArray = 0.5):
        """_summary_

        Parameters
        ----------
        vi : npt.NDArray
        vo : npt.NDArray
        frac : float | npt.NDArray
            Fraction measured from inner surface, e.g., 0.25 would be one quarter
            of the way from the inner to the outer surface. Defaults to 0.5.

        Returns
        -------
        _type_: _description_
        """
        assert isinstance(frac, (float, np.ndarray))
        frac = frac if isinstance(frac, float) else frac[:, None]
        return (1-frac) * self.white.vertices + frac * self.pial.vertices

    def place_layers(self, method: str, frac: float | npt.NDArray = 0.5):
        if method in {"equidistance", "equivolume"}:
            match method:
                case "equivolume":
                    frac = self.compute_equivolume_fraction(frac)
                case "equidistance":
                    pass
            return self._place_layer_at_distance_fraction(frac)
        elif method == "laplace":
            return

class Cortex:
    def __init__(self, lh: Hemisphere, rh: Hemisphere) -> None:
        self.lh = lh
        self.rh = rh

    def place_layers(self, frac):
        self.lh.place_layers(frac)
        self.rh.place_layers(frac)
