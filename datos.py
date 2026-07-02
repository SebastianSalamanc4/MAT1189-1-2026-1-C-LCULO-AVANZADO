"""
datos.py — Datos geográficos y de población de Temuco.

Coordenadas verificadas con OpenStreetMap (Nominatim).
Habitantes: Censo INE 2002 por macrosector (Wikipedia).
"""

import numpy as np

# Punto de referencia: Plaza de Armas de Temuco
TEMUCO_LAT_REF = -38.7359
TEMUCO_LON_REF = -72.5904

# Barrios de Temuco con coordenadas reales y población censal
BARRIOS_TEMUCO = [
    {"nombre": "Centro",            "lat": -38.7359, "lon": -72.5904, "hab": 11700},
    {"nombre": "Pueblo Nuevo",      "lat": -38.7187, "lon": -72.5613, "hab": 26043},
    {"nombre": "Pedro de Valdivia", "lat": -38.7221, "lon": -72.6171, "hab": 34490},
    {"nombre": "Santa Rosa",        "lat": -38.7395, "lon": -72.5757, "hab": 39584},
    {"nombre": "Amanecer",          "lat": -38.7563, "lon": -72.6241, "hab": 28000},
    {"nombre": "Universidad",       "lat": -38.7472, "lon": -72.6171, "hab": 27201},
    {"nombre": "Av. Alemania",      "lat": -38.7359, "lon": -72.6069, "hab": 15000},
    {"nombre": "Fundo El Carmen",   "lat": -38.7143, "lon": -72.6531, "hab": 50000},
    {"nombre": "Pueblo Nuevo Norte","lat": -38.6978, "lon": -72.5513, "hab": 12000},
]

RADIO_DETECCION_KM = 2.0


def latlon_a_km(lat, lon):
    """Convierte coordenadas geográficas a km relativos al centro de Temuco."""
    x = (lon - TEMUCO_LON_REF) * 111.32 * np.cos(np.radians(TEMUCO_LAT_REF))
    y = (lat - TEMUCO_LAT_REF) * 110.57
    return x, y


def km_a_latlon(x_km, y_km):
    """Convierte km relativos al centro de Temuco a coordenadas geográficas."""
    lon = x_km / (111.32 * np.cos(np.radians(TEMUCO_LAT_REF))) + TEMUCO_LON_REF
    lat = y_km / 110.57 + TEMUCO_LAT_REF
    return lat, lon


def detectar_barrio(lat, lon):
    """Encuentra el barrio más cercano al punto clickeado en el mapa."""
    mejor = None
    mejor_dist = float('inf')
    for d in BARRIOS_TEMUCO:
        dx = (lon - d["lon"]) * 111.32 * np.cos(np.radians(lat))
        dy = (lat - d["lat"]) * 110.57
        dist = np.sqrt(dx**2 + dy**2)
        if dist < mejor_dist:
            mejor_dist = dist
            mejor = d
    if mejor and mejor_dist < RADIO_DETECCION_KM:
        return mejor["nombre"], mejor["hab"], mejor_dist
    return None, None, mejor_dist
