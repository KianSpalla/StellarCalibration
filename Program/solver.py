import numpy as np
from scipy.spatial import cKDTree
from geometry import predict_pixels_from_catalog
"""
match_score takes a cKDTree of the image sources, the predicted pixel positions from the catalog, and a pixel tolerance.
It queries the tree to find the nearest image source for each predicted position and counts how many are within the pixel tolerance.
If minTolerance is provided, only matches with distance between minTolerance and pixelTolerance are counted.
Returns the total score (number of matches), the distances to the nearest image sources, and their indices.
"""
def match_score(imgTree, predictedXY, maxTolerance, minTolerance=None):
    starDistance, starIndex = imgTree.query(predictedXY, k=1)
    if minTolerance is not None:
        score = np.sum((starDistance >= minTolerance) & (starDistance <= maxTolerance))
    else:
        score = np.sum(starDistance <= maxTolerance)
    return score, starDistance, starIndex

"""
deduplicate_matches resolves one-to-many conflicts where multiple catalog stars match to the same detected source.
For each detected source claimed by more than one catalog star, only the closest catalog star is kept.
Returns a boolean mask over catalog indices indicating which matches survived deduplication.
"""
def deduplicate_matches(starDistance, starIndex, tolerance):
    catalogCount = len(starDistance)
    validMask = starDistance <= tolerance
    
    sourceToMatches = {}
    for catalogIndex in range(catalogCount):
        if not validMask[catalogIndex]:
            continue
        sourceIndex = int(starIndex[catalogIndex])
        if sourceIndex not in sourceToMatches:
            sourceToMatches[sourceIndex] = []
        sourceToMatches[sourceIndex].append(catalogIndex)
    
    keepMask = np.zeros(catalogCount, dtype=bool)
    for sourceIndex, catalogIndices in sourceToMatches.items():
        if len(catalogIndices) == 1:
            keepMask[catalogIndices[0]] = True
        else:
            bestCatalogIndex = catalogIndices[np.argmin(starDistance[catalogIndices])]
            keepMask[bestCatalogIndex] = True
    
    return keepMask

"""
solve_orientation takes the detected image sources, the catalog altitudes and azimuths, the center coordinates, and the radius in pixels.
It performs a search using alpha, beta, and gamma angles to find the best orientation that matches the detected sources to the catalog predictions.
It first does a coarse search over a grid of angles, then refines the search around the best solution found. 
It returns the best orientation parameters, the score, and the predicted pixel positions for the catalog stars.

NOTE: A more in depth explanation of the alpha, beta, and gamma angles can be found in comments at the bottom of this file.
"""
def solve_orientation(imgXY, catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, pixelTolerance):
    imgTree = cKDTree(imgXY)

    alphaGrid = np.deg2rad(np.arange(0, 360, 5))
    betaGrid = np.deg2rad(np.arange(0, 11, 2))
    gammaGrid = np.deg2rad(np.arange(0, 360, 20))

    best = {"score": -1}

    for beta in betaGrid:
        gammaList = [0.0] if abs(beta) < 1e-12 else gammaGrid
        for gamma in gammaList:
            for alpha in alphaGrid:
                predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
                score, starDistance, starIndex = match_score(imgTree, predictedXY, pixelTolerance)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "starDistance": starDistance,
                        "starIndex": starIndex,
                        "predictedXY": predictedXY,
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
                score, starDistance, starIndex = match_score(imgTree, predictedXY, pixelTolerance)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "starDistance": starDistance,
                        "starIndex": starIndex,
                        "predictedXY": predictedXY,
                    }

    matchedMask = best["starDistance"] <= 25.0
    matchedCount = int(np.sum(matchedMask))

    if matchedCount > 0:
        meanDist = float(np.mean(best["starDistance"][matchedMask]))
        stdDist = float(np.std(best["starDistance"][matchedMask]))
        clipUpper = meanDist + 2.0 * stdDist
        clipLower = max(0.0, meanDist - 2.0 * stdDist)

        best["score"] = -1

        alphaClip = best["alpha"] + np.deg2rad(np.arange(-5.0, 5.0 + 1e-12, 0.5))
        betaClip = best["beta"] + np.deg2rad(np.arange(-2.0, 2.0 + 1e-12, 0.2))
        gammaClip = best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0))

        alphaClip = np.mod(alphaClip, 2 * np.pi)
        gammaClip = np.mod(gammaClip, 2 * np.pi)
        betaClip = np.clip(betaClip, 0.0, np.deg2rad(15.0))

        for beta in np.unique(betaClip):
            gammaList = [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gammaClip)
            for gamma in gammaList:
                for alpha in np.unique(alphaClip):
                    predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
                    score, starDistance, starIndex = match_score(imgTree, predictedXY, maxTolerance=clipUpper, minTolerance=clipLower)
                    if score > best["score"]:
                        best = {
                            "score": score,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "starDistance": starDistance,
                            "starIndex": starIndex,
                            "predictedXY": predictedXY,
                        }

        clippedMask = (best["starDistance"] >= clipLower) & (best["starDistance"] <= clipUpper)

        # Deduplicate: resolve one-to-many matches so each detected source is claimed by at most one catalog star
        dedupMask = deduplicate_matches(best["starDistance"], best["starIndex"], clipUpper)
        finalMask = clippedMask & dedupMask

        finalCount = int(np.sum(finalMask))
        best["rms_pix"] = float(np.sqrt(np.mean(best["starDistance"][finalMask] ** 2))) if finalCount > 0 else np.nan
        best["matched_count"] = finalCount
        best["clip_upper"] = clipUpper
        best["clip_lower"] = clipLower
        best["dedup_mask"] = finalMask
    else:
        best["rms_pix"] = np.nan
        best["matched_count"] = 0
        best["dedup_mask"] = np.zeros(len(best.get("starDistance", [])), dtype=bool)

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

    PSEUDO CODE:
    Given:
      predictedXY    — predicted pixel positions for each catalog star
      starDistance[]  — distance from each predicted position to its nearest detected source
      starIndex[]    — index of the nearest detected source for each predicted position
      tolerance      — maximum match distance
      imgXY          — detected image sources (with brightness if available)

    1. Filter to valid matches:
       matches = [(catalogIdx, starIndex[catalogIdx], starDistance[catalogIdx])
                   for each catalogIdx where starDistance[catalogIdx] <= tolerance]

    2. Group by detected source index:
       groups = group matches by starIndex
           → { sourceIdx: [(catalogIdx, distance), ...] }

    3. For each group with more than one match (conflict):
           candidates = all (catalogIdx, distance) pairs claiming this source

           Pick the winner:
             — Option A (distance priority): keep the candidate with the smallest distance
             — Option B (brightness priority): keep the catalog star with the highest brightness
             — Option C (combined):  weight by both distance and brightness to pick best

           Remove all losing candidates from the match list

    4. Build final one-to-one match arrays:
       matchedCatalogIdxs  = [winner catalogIdx for each unique source]
       matchedSourceIdxs   = [corresponding source index]
       matchedDistances    = [corresponding distances]

    5. Recompute score = len(matchedCatalogIdxs)

    Return updated score, matchedCatalogIdxs, matchedSourceIdxs, matchedDistances

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