import os
import csv
from astroquery.simbad import Simbad

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMED_STARS_FILE = os.path.join(CACHE_DIR, "named_stars.csv")

"""
Queries SIMBAD for all stars brighter than mag_limit (V band).
Cross-references with SIMBAD's identifier table to find common names (entries starting with "NAME ").
Stars without a common name get an empty name field.
Saves to CSV with columns: name, ra, dec, vmag
"""
def build_named_star_cache(mag_limit=6.5):
    print(f"Querying SIMBAD for stars with V < {mag_limit} ...")
    stars = Simbad.query_tap(f"""
    SELECT b.oid, b.main_id, b.ra, b.dec, f.flux AS vmag
    FROM basic AS b
    JOIN flux AS f ON b.oid = f.oidref
    WHERE f.filter = 'V'
      AND f.flux IS NOT NULL
      AND f.flux < {mag_limit}
      AND b.ra IS NOT NULL
      AND b.dec IS NOT NULL
    ORDER BY vmag ASC
    """)
    print(f"Retrieved {len(stars)} stars.")

    # Collect all oids to query for common names
    oid_list = [str(row["oid"]) for row in stars]

    # SIMBAD TAP has an IN clause limit, so batch in chunks
    name_map = {}
    batch_size = 500
    for i in range(0, len(oid_list), batch_size):
        batch = oid_list[i : i + batch_size]
        oids_str = ",".join(batch)
        names = Simbad.query_tap(f"""
        SELECT i.oidref, i.id
        FROM ident AS i
        WHERE i.oidref IN ({oids_str})
          AND i.id LIKE 'NAME %'
        """)
        if names is not None:
            for row in names:
                oid = int(row["oidref"])
                raw_name = str(row["id"])
                clean_name = raw_name[5:] if raw_name.startswith("NAME ") else raw_name
                # Keep the first name found per star (skip duplicates)
                if oid not in name_map:
                    name_map[oid] = clean_name

    # Write CSV
    with open(NAMED_STARS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "ra", "dec", "vmag"])
        for row in stars:
            oid = int(row["oid"])
            name = name_map.get(oid, "")
            writer.writerow([name, f"{float(row['ra']):.10f}", f"{float(row['dec']):.10f}", f"{float(row['vmag']):.4f}"])

    named_count = sum(1 for oid in [int(r["oid"]) for r in stars] if oid in name_map)
    print(f"Saved {len(stars)} stars ({named_count} named) to {NAMED_STARS_FILE}")


if __name__ == "__main__":
    build_named_star_cache()
