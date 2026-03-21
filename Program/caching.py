import os
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astroquery.gaia import Gaia

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(CACHE_DIR, "cache.csv")


def build_cache(gmax=6.5):
    Gaia.ROW_LIMIT = -1

    query = f"""
    SELECT ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag IS NOT NULL
      AND phot_g_mean_mag < {gmax}
    ORDER BY phot_g_mean_mag ASC
    """

    print(f"Querying Gaia DR3 for stars with G < {gmax} ...")
    job = Gaia.launch_job_async(query)
    tbl = job.get_results()
    print(f"Retrieved {len(tbl)} stars.")

    ra = np.array(tbl["ra"], dtype=np.float64)
    dec = np.array(tbl["dec"], dtype=np.float64)
    gmag = np.array(tbl["phot_g_mean_mag"], dtype=np.float32)

    header = "ra,dec,phot_g_mean_mag"
    data = np.column_stack((ra, dec, gmag))
    np.savetxt(CACHE_FILE, data, delimiter=",", header=header, comments="", fmt="%.10f")
    print(f"Cache saved to {CACHE_FILE} ({len(ra)} stars).")


def load_cache():
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache not found at {CACHE_FILE}. Run build_cache() first.")
    data = np.loadtxt(CACHE_FILE, delimiter=",", skiprows=1)
    ra = data[:, 0]
    dec = data[:, 1]
    gmag = data[:, 2]
    return ra, dec, gmag


def filter_cache_by_location(meta, gmax=2.5, catalogRadiusDeg = 60.0):
    ra, dec, gmag = load_cache()
    bright = gmag < gmax
    ra, dec, gmag = ra[bright], dec[bright], gmag[bright]

    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m = float(meta["GPS"]["altitude"])
    ut_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    obstime = Time(ut_iso, scale="utc")

    stars_icrs = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    stars_altaz = stars_icrs.transform_to(AltAz(obstime=obstime, location=location))

    alt = stars_altaz.alt.deg
    az = stars_altaz.az.deg

    above = alt > (90 - catalogRadiusDeg)
    alt, az, gmag = alt[above], az[above], gmag[above]

    return alt, az, gmag


if __name__ == "__main__":
    build_cache()
