from .city_grid import CityGrid, LAND_BUILDING, LAND_EMPTY, LAND_GREEN, LAND_ROAD

try:
    import scipy  # noqa: F401
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if SCIPY_AVAILABLE:
    from .terrain import TERRAIN_ARCHETYPES, generate_terrain
    from .buildings import CITY_ARCHETYPES, generate_buildings
else:
    TERRAIN_ARCHETYPES = ()
    CITY_ARCHETYPES = ()
    generate_terrain = None
    generate_buildings = None

__all__ = [
    "CityGrid",
    "LAND_BUILDING",
    "LAND_EMPTY",
    "LAND_GREEN",
    "LAND_ROAD",
    "TERRAIN_ARCHETYPES",
    "generate_terrain",
    "CITY_ARCHETYPES",
    "generate_buildings",
]
