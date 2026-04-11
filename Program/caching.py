import os
import csv
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from planets import get_planets

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(CACHE_DIR, "stars.csv")


def radec_to_altaz(ra_deg, dec_deg, obstime, location):
    icrs = SkyCoord(ra=np.asarray(ra_deg, dtype=float) * u.deg, dec=np.asarray(dec_deg, dtype=float) * u.deg, frame="icrs")
    altaz = icrs.transform_to(AltAz(obstime=obstime, location=location))
    return np.asarray(altaz.alt.deg, dtype=float), np.asarray(altaz.az.deg, dtype=float)


def load_cache():
    names = []
    ra_list = []
    dec_list = []
    mag_list = []
    with open(CACHE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
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

    alt, az = radec_to_altaz(ra, dec, obstime, location)

    above = alt > (90 - catalogRadiusDeg)
    alt, az, mag = alt[above], az[above], mag[above]
    names = [n for n, a in zip(names, above) if a]


    planet_data = get_planets(meta)

    planet_ra = []
    planet_dec = []
    planet_mag = []
    planet_names = []

    for planet_name, vals in planet_data.items():
        planet_ra.append(float(vals["ra_hours"]) * 15.0)
        planet_dec.append(float(vals["dec_degrees"]))
        brightness_mag = float(vals["brightness_mag"])
        planet_mag.append(brightness_mag)
        planet_names.append(planet_name)

    if planet_ra:
        planet_alt, planet_az = radec_to_altaz(planet_ra, planet_dec, obstime, location)
        visible = planet_alt > (90.0 - float(catalogRadiusDeg))

        if np.any(visible):
            alt = np.concatenate([np.asarray(alt, dtype=float), planet_alt[visible]])
            az = np.concatenate([np.asarray(az, dtype=float), planet_az[visible]])
            mag = np.concatenate([np.asarray(mag, dtype=float), np.asarray(planet_mag, dtype=float)[visible]])
            names = list(names) + [n for n, is_visible in zip(planet_names, visible) if is_visible]

    return alt, az, mag, names, planet_data
