import os
import csv
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(CACHE_DIR, "stars.csv")


def load_cache():
    names = []
    ra_list = []
    dec_list = []
    mag_list = []
    with open(CACHE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            names.append(row[0])
            ra_list.append(float(row[1]))
            dec_list.append(float(row[2]))
            mag_list.append(float(row[3]))
    return np.array(ra_list), np.array(dec_list), np.array(mag_list), names


def filter_cache_by_location(meta, gmax=2.5, catalogRadiusDeg = 60.0):
    ra, dec, mag, names = load_cache()
    bright = mag < gmax
    ra, dec, mag = ra[bright], dec[bright], mag[bright]
    names = [n for n, b in zip(names, bright) if b]

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
    alt, az, mag = alt[above], az[above], mag[above]
    names = [n for n, a in zip(names, above) if a]

    return alt, az, mag, names
