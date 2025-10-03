# Configuración de la API QBikes

# Puerto del servidor
PORT = 8000
HOST = "0.0.0.0"

# Configuración de OSMnx
OSMNX_CACHE_FOLDER = "./osmnx_cache"
OSMNX_USE_CACHE = True

# Configuración de POIs
POI_RADIUS_METERS = 100  # Radio para considerar un POI como "cubierto"

# Configuración de optimización
DEFAULT_LAMBDA_COST = 1.0
POI_WEIGHT_MULTIPLIER = 10  # Multiplicador para dar más importancia a POIs en la heurística

# Configuración de timeouts (segundos)
OSM_DOWNLOAD_TIMEOUT = 300

# Configuración de logging
LOG_LEVEL = "INFO"

# Atributos OSM para detectar ciclovías
BIKEWAY_ATTRIBUTES = {
    'highway': ['cycleway'],
    'cycleway': ['lane', 'track', 'opposite', 'opposite_lane', 'opposite_track', 'shared_lane'],
    'cycleway:right': ['lane', 'track', 'shared_lane'],
    'cycleway:left': ['lane', 'track', 'shared_lane'],
    'cycleway:both': ['lane', 'track', 'shared_lane']
}
