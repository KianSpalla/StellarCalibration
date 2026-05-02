import numpy as np
import time
from GONet_Wizard.GONet_utils import GONetFile
from detection import dynamic_find_stars, find_centroids, filter_by_size
from caching import filter_cache_by_location
from geometry import filter_image_sources_by_radius
from solver import solve_orientation
from centering import find_zenith_pixel_and_center, build_shifted_image
from helpers import rotate_image

"""
MAIN PIPELINE
"""
def run_calibration(
    imagePath,
    show_plots=False,
    vmag=2.5,
    pixelMin=5,
    pixelMax=50,
    catalogRadiusDeg=60.0,
    sectionSize=200,
):
#Pipeline
    #Center pixels and radius. This may change based on what GONet is being used
    cx, cy = 1030, 760
    radiusPix = 740

    # Start timer for testing
    t0 = time.perf_counter()

    #Get GONet image from path
    go = GONetFile.from_file(imagePath)
    go.remove_overscan()

    #Get the green channel in img
    img = go.green

    #Rotation is for testing only. This should be set to 0 in production
    rotationAngle = 0
    img = rotate_image(img, rotationAngle, cx, cy)

    #Threshold for finding stars
    N = 5

    #Finds stars in image
    labels, numLabels = dynamic_find_stars(img, N, sectionSize)

    #Filters stars based on size 
    labels, numLabels = filter_by_size(labels, numLabels, pixelMin, pixelMax)

    #Finds centroids and fluxes for each star
    xCentroids, yCentroids, totalFluxes = find_centroids(img, labels, numLabels)
    imgXY = np.column_stack([xCentroids, yCentroids])

    #filters out stars based on radius
    imgXY = filter_image_sources_by_radius(imgXY=imgXY, cx=cx, cy=cy, radiusPix=radiusPix, radiusDeg = catalogRadiusDeg)

    #Grab meta data from GONet Image
    meta = go.meta
    #Grabs stars + planets from cache that appear in the sky at the date, time, and location from the meta data.
    catalogAltDeg, catalogAzDeg, catalogVmag, catalogNames, planet_data = filter_cache_by_location(meta, vmag=vmag, catalogRadiusDeg=catalogRadiusDeg)

    #Solver that rotates through three angles and gets matches, returns the best rotations
    best = solve_orientation(imgXY, catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, 35, catalogVmag, totalFluxes)

    #Get zenith pixel coordinates based on best solve
    centerResult = find_zenith_pixel_and_center(img=img, best=best, cx=cx, cy=cy, radiusPix=radiusPix)

#Formating
    print(f"Time for pipeline={time.perf_counter() - t0:.3f}")

    print(f"catalog_stars={len(catalogAltDeg)}, image_sources={len(imgXY)}")
    print(f"score={best['score']:.2f}, matched={best['matched_count']}, rms_pix={best['rms_pix']:.3f}")
    print(
        "alpha_deg={:.3f}, beta_deg={:.3f}, gamma_deg={:.3f}".format(
            np.rad2deg(best["alpha"]),
            np.rad2deg(best["beta"]),
            np.rad2deg(best["gamma"]),
        )
    )
    print(f"Zenith pixel: x={centerResult['zenithX']:.2f}, y={centerResult['zenithY']:.2f}")
    print(f"Applied shift: dx={centerResult['shiftX']:.2f}, dy={centerResult['shiftY']:.2f}")

    if show_plots:
        import matplotlib.pyplot as plt

        imgMean = float(np.mean(img))
        imgStd = float(np.std(img))

        fig1, ax1 = plt.subplots()
        ax1.imshow(img, origin="upper", cmap="gray",
                   vmin=imgMean - 2 * imgStd, vmax=imgMean + 5 * imgStd)
        ax1.scatter(imgXY[:, 0], imgXY[:, 1], s=50, edgecolor="red",
                    facecolor="none", label="Detected sources")
        ax1.scatter(best["predictedXY"][:, 0], best["predictedXY"][:, 1], s=50,
                    edgecolor="blue", facecolor="none", label="Catalog predictions")

        matchedCatalogIdx = best.get("matched_catalog_indices", best.get("matched_catallog_indicies", np.array([], dtype=int)))
        matchedImageIdx = best.get("matched_image_indices", best.get("matched_image_indicies", np.array([], dtype=int)))
        matchedCatalogSet = set(np.asarray(matchedCatalogIdx, dtype=int))

        for catIdx, srcIdx in zip(np.asarray(matchedCatalogIdx, dtype=int), np.asarray(matchedImageIdx, dtype=int)):
            if 0 <= catIdx < len(best["predictedXY"]) and 0 <= srcIdx < len(imgXY):
                catX, catY = best["predictedXY"][catIdx]
                srcX, srcY = imgXY[srcIdx]
                ax1.plot([catX, srcX], [catY, srcY], color="lime", linewidth=0.8, alpha=0.7)
        ax1.plot([], [], color="lime", linewidth=0.8, label="Matched pairs")

        for catIdx in range(len(best["predictedXY"])):
            if catIdx in matchedCatalogSet and catIdx < len(catalogNames) and catalogNames[catIdx]:
                px, py = best["predictedXY"][catIdx]
                ax1.annotate(catalogNames[catIdx], (px, py), color="white",
                             fontsize=7, ha="left", va="bottom",
                             xytext=(4, 4), textcoords="offset points")

        ax1.scatter([centerResult["targetCenterX"]], [centerResult["targetCenterY"]],
                    s=100, marker="+", c="yellow", label="Image centre")
        ax1.scatter([centerResult["zenithX"]], [centerResult["zenithY"]],
                    s=120, marker="x", c="cyan", label="Zenith")
        ax1.plot(
            [centerResult["zenithX"], centerResult["targetCenterX"]],
            [centerResult["zenithY"], centerResult["targetCenterY"]],
            color="cyan", linestyle="--", linewidth=1.5, label="Applied shift",
        )
        ax1.legend()
        ax1.set_title(f"Orientation solve \u2014 score: {best['score']} matches")
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.imshow(centerResult["centeredSub"], origin="upper", cmap="gray",
                   vmin=imgMean - 2 * imgStd, vmax=imgMean + 5 * imgStd)
        ax2.scatter([centerResult["targetCenterX"]], [centerResult["targetCenterY"]],
                    s=120, marker="x", c="cyan", label="Zenith (centred)")
        ax2.legend()
        ax2.set_title("Shifted image \u2014 zenith at centre")
        plt.show()

    shiftedResult = build_shifted_image(
        imagePath=imagePath,
        shiftX=centerResult["shiftX"],
        shiftY=centerResult["shiftY"],
        alphaDeg=centerResult["alphaDeg"] - rotationAngle,
    )
    print("Shifted image prepared (not yet saved).")

    return {
        "best": best,
        "centerResult": centerResult,
        "img": img,
        "imgXY": imgXY,
        "catalogNames": catalogNames,
        "planetData": planet_data,
        "planet_data": planet_data,
        "shiftedImage": shiftedResult,
        "shifted_image": shiftedResult,
        "shiftedFormat": "PNG",
        "shifted_format": "PNG",
        "suggested_suffix": ".png",
    }

if __name__ == "__main__":
    run_calibration(r"C:\Users\spall\Desktop\GONet\StellarCalibration\Testing Images\202_250628_063009_1751092241.jpg", show_plots=True, vmag=2.5)