import numpy as np
from scipy.spatial import cKDTree
from GONet_Wizard.GONet_utils import GONetFile
from pathlib import Path
import matplotlib.pyplot as plt
from astrometry_1 import *
from centroid_recap import *
from GoNet_Exercise_DecemberSixth import *
# ----------------------------
# 1) Small math helpers
# ----------------------------
def unitvec_from_altaz(alt_rad, az_rad):
    """
    Convert (alt, az) to a unit vector in the local sky frame.
    Convention used here:
      x = east, y = north, z = up
    """
    ca = np.cos(alt_rad)
    x = ca * np.sin(az_rad)   # east
    y = ca * np.cos(az_rad)   # north
    z = np.sin(alt_rad)       # up
    return np.stack([x, y, z], axis=-1)


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], float)


def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], float)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], float)


def orientation_matrix(alpha, beta, gamma):
    """
    Build R that maps camera vectors -> sky vectors.

    Parameters
    ----------
    alpha : float
        Rotation about camera optical axis (azimuth zero-point), radians.
    beta : float
        Tilt magnitude, radians.
    gamma : float
        Tilt direction (azimuth in sky frame), radians.

    Intern-friendly interpretation:
      - alpha spins the camera around its own axis.
      - beta and gamma tilt the camera axis away from zenith.

    TODO:
      Decide and document rotation order carefully.
      One workable choice:
        1) spin camera around its axis by alpha (in camera frame)
        2) tilt the axis by beta toward direction gamma (in sky frame)
    """
    # One reasonable construction:
    # tilt toward azimuth gamma can be done by:
    #   rotate sky frame so tilt direction aligns with +y, tilt about x or y, rotate back
    # Keep it simple: use Z(gamma) then X(beta) then Z(-gamma), combined with alpha.

    R_alpha = rot_z(alpha)  # spin about camera z (if we treat cam z as "up" initially)
    R_tilt  = rot_z(gamma) @ rot_x(beta) @ rot_z(-gamma)

    # Map camera -> sky:
    R = R_tilt @ R_alpha
    return R


# ----------------------------
# 2) Lens model (simple)
# ----------------------------
def theta_from_r(r_pix, R_pix):
    """Simple equidistant mapping: R_pix -> 90 deg."""
    # theta in radians
    return (r_pix / R_pix) * (np.pi / 2)

def r_from_theta(theta_rad, R_pix):
    """Inverse of theta_from_r."""
    return (theta_rad / (np.pi / 2)) * R_pix


# ----------------------------
# 3) Forward model: catalog Alt/Az -> predicted pixels
# ----------------------------
def predict_pixels_from_catalog(alt_deg, az_deg, cx, cy, R_pix, alpha, beta, gamma):
    alt = np.deg2rad(alt_deg)
    az  = np.deg2rad(az_deg)

    v_sky = unitvec_from_altaz(alt, az)   # (M,3)

    R = orientation_matrix(alpha, beta, gamma)   # cam -> sky
    # We need sky -> cam, so invert (rotation matrix inverse = transpose)
    v_cam = v_sky @ R.T

    # Convert cam vector -> (theta, phi)
    # theta = angle from cam +z
    z = np.clip(v_cam[:, 2], -1, 1)
    theta = np.arccos(z)
    phi = np.arctan2(v_cam[:, 1], v_cam[:, 0])

    # Convert theta -> pixel radius
    r = r_from_theta(theta, R_pix)

    # Polar -> pixel
    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)
    return np.stack([x, y], axis=-1)


# ----------------------------
# 4) Matching score
# ----------------------------
def match_score(img_xy, pred_xy, tol_pix=20.0):
    tree = cKDTree(img_xy)
    dists, idx = tree.query(pred_xy, k=1)
    score = np.sum(dists <= tol_pix)
    return int(score), dists, idx


# ----------------------------
# 5) Optimization loop (coarse grid + refine)
# ----------------------------
def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix):
    """
    Solve for alpha, beta, gamma by maximizing number of matches.

    Suggested search ranges (start easy):
      alpha: 0..360 deg
      beta : 0..10 deg (camera should be close to zenith)
      gamma: 0..360 deg (only matters if beta>0)
    """
    # Coarse grid (tune for runtime)
    alpha_grid = np.deg2rad(np.arange(0, 360, 5))
    #print( alpha_grid)
    beta_grid  = np.deg2rad(np.arange(0, 11, 2))     # 0,2,4,6,8,10 deg
    gamma_grid = np.deg2rad(np.arange(0, 360, 20))   # coarse; refine later

    best = {"score": -1}

    for beta in beta_grid:
        # If beta is 0, gamma doesn’t matter — you can skip gamma loop for speed
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid

        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg,
                    cx, cy, R_pix,
                    alpha, beta, gamma
                )
                s, dists, idx = match_score(img_xy, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "dists": dists,
                        "idx": idx,
                        "pred_xy": pred_xy
                    }

    # TODO (refinement):
    # After coarse best, search in a small box around it with smaller steps.
    # e.g., alpha ± 5° step 0.5°, beta ± 2° step 0.2°, gamma ± 10° step 1°

    return best

def main():
    go = GONetFile.from_file(r"Testing Images\256_251029_204008_1761770474.jpg")
    go.remove_overscan()
    N = 5
    go_ravel=np.ravel(go.green)
    sub = go.green
    sub_mean=go_ravel.mean()
    sub_std=go_ravel.std()
    threshold = sub_mean+N*sub_std
    mask = sub>threshold
    labels,num_labels=compact_search(np.array(mask))
    sources,x_centroids,y_centroids=measure_sources(sub, labels, num_labels)
    img_xy = np.column_stack([x_centroids, y_centroids])
    cx, cy = 1030, 760
    R_pix=740
    cat_az_deg=az
    cat_alt_deg=alt
    best=solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix)
    print(best)

if __name__ == "__main__":
    main()