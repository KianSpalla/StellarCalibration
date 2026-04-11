import numpy as np
from scipy.spatial import cKDTree
from geometry import predict_pixels_from_catalog

def percentile_ranks(values, larger_is_brighter=True):
    """
    Convert an array of brightness-like values into percentile ranks in [0, 1].
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    if larger_is_brighter:
        order = order[::-1]

    ranks = np.empty_like(values, dtype=float)
    if len(values) == 1:
        ranks[order[0]] = 1.0
        return ranks

    ranks[order] = np.linspace(1.0, 0.0, len(values))
    return ranks


def candidate_match_cost(distance, img_rank, cat_rank, pixel_tolerance, lam=0.2):
    """
    Combined geometric + brightness-rank cost.

    Lower is better.
    """
    return (distance / pixel_tolerance) + lam * abs(img_rank - cat_rank)


def match_with_local_brightness_tiebreak(img_xy, predicted_xy, source_fluxes, catalog_mags, pixel_tolerance=25.0, k_neighbors=5, lam=0.2,):
    """
    Match predicted catalog positions to detected sources using:
      1) geometry first
      2) brightness-rank as a local tiebreaker

    Intended use:
    - predicted_xy comes from predict_pixels_from_catalog(...)
    - source_fluxes should be aligned with img_xy
    - catalog_mags should be aligned with predicted_xy

    """
    img_xy = np.asarray(img_xy, dtype=float)
    predicted_xy = np.asarray(predicted_xy, dtype=float)
    source_fluxes = np.asarray(source_fluxes, dtype=float)
    catalog_mags = np.asarray(catalog_mags, dtype=float)

    tree = cKDTree(img_xy)

    # Brightest image source -> rank near 1
    img_ranks = percentile_ranks(source_fluxes, larger_is_brighter=True)

    # Smallest magnitude -> brightest -> rank near 1
    cat_ranks = percentile_ranks(catalog_mags, larger_is_brighter=False)

    # Query a few nearest detections for each predicted catalog point
    dists_all, idx_all = tree.query(predicted_xy, k=k_neighbors)

    if k_neighbors == 1:
        dists_all = dists_all[:, None]
        idx_all = idx_all[:, None]

    chosen_img_idx = np.full(len(predicted_xy), -1, dtype=int)
    chosen_dist = np.full(len(predicted_xy), np.inf, dtype=float)

    for i in range(len(predicted_xy)):
        dists = dists_all[i]
        idxs = idx_all[i]

        # Consider only geometrically plausible candidates
        good = dists <= pixel_tolerance
        if not np.any(good):
            continue

        dists = dists[good]
        idxs = idxs[good]

        # If nearest is clearly better than second nearest, trust geometry
        if len(dists) == 1 or dists[0] < 0.6 * dists[1]:
            chosen_img_idx[i] = int(idxs[0])
            chosen_dist[i] = float(dists[0])
            continue

        # Otherwise use combined cost as a tie-breaker
        costs = [
            candidate_match_cost(
                distance=float(d),
                img_rank=float(img_ranks[j]),
                cat_rank=float(cat_ranks[i]),
                pixel_tolerance=pixel_tolerance,
                lam=lam,
            )
            for d, j in zip(dists, idxs)
        ]
        best_local = int(np.argmin(costs))
        chosen_img_idx[i] = int(idxs[best_local])
        chosen_dist[i] = float(dists[best_local])

    # Deduplicate: keep the closest catalog star for each image source
    best_for_image = {}
    for cat_i, (img_i, dist) in enumerate(zip(chosen_img_idx, chosen_dist)):
        if img_i < 0 or not np.isfinite(dist):
            continue
        if img_i not in best_for_image or dist < best_for_image[img_i][0]:
            best_for_image[img_i] = (dist, cat_i)

    keep_cat_idx = np.array([v[1] for v in best_for_image.values()], dtype=int)
    keep_img_idx = chosen_img_idx[keep_cat_idx]
    keep_dists = chosen_dist[keep_cat_idx]

    score = int(np.sum(keep_dists <= pixel_tolerance))
    rms_pix = float(np.sqrt(np.mean(keep_dists**2))) if len(keep_dists) > 0 else np.nan

    return score, keep_cat_idx, keep_img_idx, keep_dists, rms_pix
        #{
        # "score": score,
        # "matched_catalog_indices": keep_cat_idx,
        # "matched_image_indices": keep_img_idx,
        # "distances_pix": keep_dists,
        # "rms_pix": rms_pix,
        #}
        


    

"""
solve_orientation takes the detected image sources, the catalog altitudes and azimuths, the center coordinates, and the radius in pixels.
It performs a search using alpha, beta, and gamma angles to find the best orientation that matches the detected sources to the catalog predictions.
It first does a coarse search over a grid of angles, then refines the search around the best solution found. 
It returns the best orientation parameters, the score, and the predicted pixel positions for the catalog stars.

NOTE: A more in depth explanation of the alpha, beta, and gamma angles can be found in comments at the bottom of this file.
"""
def solve_orientation(imgXY, catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, pixelTolerance, catalogMag, sourceFluxes):
    #imgTree = cKDTree(imgXY)

    alphaGrid = np.deg2rad(np.arange(0, 360, 5))
    betaGrid = np.deg2rad(np.arange(0, 11, 2))
    gammaGrid = np.deg2rad(np.arange(0, 360, 20))

    best = {"score": -1}

    for beta in betaGrid:
        gammaList = [0.0] if abs(beta) < 1e-12 else gammaGrid
        for gamma in gammaList:
            for alpha in alphaGrid:
                predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
                score, matched_catalog_indices, matched_image_indices, distances_pix, rms_pix = match_with_local_brightness_tiebreak(imgXY, predictedXY, sourceFluxes, catalogMag, pixelTolerance)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "distances_pix": distances_pix,
                        "matched_catallog_indicies": matched_catalog_indices,
                        "matched_image_indicies": matched_image_indices,
                        "predictedXY": predictedXY,
                        "rms_pix" : rms_pix
                    }

    alphaRefine = best["alpha"] + np.deg2rad(np.arange(-5.0, 5.0 + 1e-12, 0.5))
    betaRefine = best["beta"] + np.deg2rad(np.arange(-2.0, 2.0 + 1e-12, 0.2))
    gammaRefine = best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0))

    alphaRefine = np.mod(alphaRefine, 2 * np.pi)
    gammaRefine = np.mod(gammaRefine, 2 * np.pi)
    betaRefine = np.clip(betaRefine, 0.0, np.deg2rad(15.0))

    for beta in np.unique(betaRefine):
        gammaList = [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gammaRefine)
        for gamma in gammaList:
            for alpha in np.unique(alphaRefine):
                predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
                score, matched_catalog_indices, matched_image_indices, distances_pix, rms_pix = match_with_local_brightness_tiebreak(imgXY, predictedXY, sourceFluxes, catalogMag, pixelTolerance)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "distances_pix": distances_pix,
                        "matched_catallog_indicies": matched_catalog_indices,
                        "matched_image_indicies": matched_image_indices,
                        "predictedXY": predictedXY,
                        "rms_pix" : rms_pix
                    }
    best["matched_count"] = len(matched_image_indices)

    # matchedMask = best["distances_pix"] <= 25.0
    # matchedCount = int(np.sum(matchedMask))

    # if matchedCount > 0:
    #     medianDist = np.median(best["distances_pix"][matchedMask])
    #     madDist = np.std(best["distances_pix"][matchedMask] / medianDist)
    #     clipUpper = medianDist + 3 * 1.4826 * madDist

    #     best["score"] = -1

    #     alphaClip = best["alpha"] + np.deg2rad(np.arange(-5.0, 5.0 + 1e-12, 0.5))
    #     betaClip = best["beta"] + np.deg2rad(np.arange(-2.0, 2.0 + 1e-12, 0.2))
    #     gammaClip = best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0))

    #     alphaClip = np.mod(alphaClip, 2 * np.pi)
    #     gammaClip = np.mod(gammaClip, 2 * np.pi)
    #     betaClip = np.clip(betaClip, 0.0, np.deg2rad(15.0))

    #     for beta in np.unique(betaClip):
    #         gammaList = [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gammaClip)
    #         for gamma in gammaList:
    #             for alpha in np.unique(alphaClip):
    #                 predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
    #                 score, matched_catalog_indices, matched_image_indices, distances_pix, rms_pix = match_with_local_brightness_tiebreak(imgXY, predictedXY, sourceFluxes, catalogMag, pixelTolerance)
    #                 if score > best["score"]:
    #                     best = {
    #                     "score": score,
    #                     "alpha": alpha,
    #                     "beta": beta,
    #                     "gamma": gamma,
    #                     "distances_pix": distances_pix,
    #                     "matched_catallog_indicies": matched_catalog_indices,
    #                     "matched_image_indicies": matched_image_indices,
    #                     "predictedXY": predictedXY,
    #                     "rms_pix" : rms_pix
    #                     }

    #     # clippedMask = (best["distances_pix"] >= clipLower) & (best["distances_pix"] <= clipUpper)
    #     # finalMask = clippedMask

    #     #finalMask = clippedMask

    #     # finalCount = int(np.sum(finalMask))
    #     # best["rms_pix"] = float(np.sqrt(np.mean(best["distances_pix"][finalMask] ** 2))) if finalCount > 0 else np.nan
    #     best["matched_count"] = score
    #     # best["clip_upper"] = clipUpper
    #     # best["clip_lower"] = clipLower
    #     # best["final_mask"] = finalMask
    # else:
    #     best["matched_count"] = score

    return best

"""
TODO:

1. Clipping stars based on average distance. (DONE)
    After the refinement pass, we compute mean + 2 * std of matched distances as a clip tolerance,
    then re-run a third refinement pass using that tighter tolerance so outliers can't influence alpha/beta/gamma.

2. Gaia cache (DONE)
    Create a cache of stars gathered from the gaia database, so that we can use the cache when searching for stars instead
    of calling gaia everytime, this will improve performance, as well solving the instance of gaia being down
    We would give priority to the cache for searching and if necessary use gaia as a fallback.

3. Ensuring we dont get multiple stars matched to the same source (DONE)
    Create a flag for star indexes on wether they are matches or not
    (This could probably be a int based on how many matches each star gets, so ideal = 1)
    (Go from one-to-many -> one-to-one relationships)
    give priority to stars based on brightness and or distance.
"""

"""
ALPHA, BETA, GAMMA EXPLANATION

To understand the Alpha, Beta, and Gamma values, it is important to fully understand the problem we are facing. 
This problem is the incorrect placement of GONet Cameras. There are two conditions for a GONet to be considered "Calibrated"
    1. The GONet camera must be facing directly north
    2. The GONet camera must be perfectly level

Alpha, Beta, and Gamma tie directly to these two issues.
    Rule 1 is solved through Alpha
    Rule 2 is solved by Beta and Gamma in unison

As there are two rules for a GONet being calibrated, every solution involves two steps
    1. Rotating the camera so that North is centered at the top (Alpha)
    2. Tilting the camera so that the zenith is at the center of the image (Beta + Gamma)

Alpha is independent of Beta + Gamma
-Alpha
    Alpha is the rotation about the optical axis. In the sense of correcting GONet orientation, 
    this would be pointing the arrows on a GONet so that they are directly north

Beta + Gamma are tied together
-Beta
    Beta is the tilt away from level. In the sense of correcting GONet orientation, 
    this would be how much the camera is tipped.
-Gamma
    Gamma is the direction of the tilt. In the sense of correcting GONet orientation, 
    this would be which cardinal direction (North, East, West, South, or anything in between) the tilt points towards

This is why when beta is 0, gamma is irrelevant, as if there is no tilt away from level,
it doesn't matter which direction the nonexistent tilt points towards

When we are solving for this problem in the solve_orientation function, we are searching over these Alpha, Beta, and Gamma values
and looking for the combination of these three values that gives us the most amount of matches.
"""