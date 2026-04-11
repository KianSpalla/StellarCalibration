from datetime import datetime
from pathlib import Path

from skyfield.api import load


# Fixed representative visual magnitudes (constant brightness values).
PLANET_CONSTANT_MAG = {
    "Mercury": -0.4,
    "Venus": -4.4,
    "Mars": -1.5,
    "Jupiter": -2.7,
    "Saturn": 0.5,
}


def get_planets(meta):
    date_time_str = meta["DateTime"]

    dt = datetime.strptime(date_time_str.strip(), "%Y:%m:%d %H:%M:%S")

    eph_path = Path(__file__).with_name("de421.bsp")


    ts = load.timescale(builtin=True)
    eph = load(str(eph_path))
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    earth = eph["earth"]

    skyfield_names = {
        "Mercury": "mercury",
        "Venus": "venus",
        "Mars": "mars",
        "Jupiter": "jupiter barycenter",
        "Saturn": "saturn barycenter",
    }

    results = {}
    for planet_name, skyfield_key in skyfield_names.items():
        astrometric = earth.at(t).observe(eph[skyfield_key])
        ra, dec, _ = astrometric.radec()

        results[planet_name] = {
            "ra_hours": float(ra.hours),
            "dec_degrees": float(dec.degrees),
            "brightness_mag": float(PLANET_CONSTANT_MAG[planet_name]),
        }

    return results