from skyfield.api import load

PLANET_CONSTANT_MAG = {
    "Mercury": -0.4,
    "Venus": -4.4,
    "Mars": -1.5,
    "Jupiter": -2.7,
    "Saturn": 0.5,
}

ts = load.timescale()
t = ts.utc(1980, 1, 1)
planets = load('de421.bsp')
mercury, venus, earth, mars, jupiter, saturn = planets['mercury'], planets['venus'], planets['earth'], planets['mars'], planets['jupiter'], planets['saturn']

d = earth.at(t).observe(mars).apparent().distance()

print('Mars is {:.2f} au from Earth'.format(d.au))