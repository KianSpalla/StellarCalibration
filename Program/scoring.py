import numpy as np
from scipy.spatial import cKDTree

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


def match_with_local_brightness_tiebreak(img_xy, tree, predicted_xy, source_fluxes, catalog_mags, pixel_tolerance=25.0, k_neighbors=5, lam=0.2,):
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

    #tree = cKDTree(img_xy)

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