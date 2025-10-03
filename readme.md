# QBikes Backend - API de Optimización de Ciclovías

API en FastAPI para optimizar redes de ciclovías usando datos de OpenStreetMap.

## Características

- 🚴 Obtención de grafos de calles desde OpenStreetMap usando `osmnx`
- 🗺️ Identificación automática de ciclovías existentes
- 🎯 Optimización heurística para seleccionar nuevas ciclovías
- 📍 Soporte para puntos de interés (POIs) con pesos
- 📊 Documentación Swagger automática en `/docs`
- ✅ Health check endpoint para monitoreo

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar servidor
python run.py
```

Luego abre http://localhost:8000/docs y prueba la API interactivamente.

## API Endpoints

- **GET** `/health` - Health check
- **POST** `/optimize_bikeway` - Optimizar red de ciclovías
- **GET** `/docs` - Documentación Swagger interactiva

## Ejemplo de Uso

**Request:**
```json
{
  "city": "Piedmont, California, USA",
  "poi": [
    {
      "lat": 37.8244,
      "lon": -122.2315,
      "weight": 10.0
    },
    {
      "lat": 37.8254,
      "lon": -122.2325,
      "weight": 5.0
    }
  ],
  "budget_km": 5.0,
  "lambda_cost": 1.0
}
```

**Response:**
```json
{
  "selected_edges": [
    {
      "start_lat": 37.8244,
      "start_lon": -122.2315,
      "end_lat": 37.8254,
      "end_lon": -122.2325,
      "length_m": 150.5,
      "is_existing": false
    }
  ],
  "total_length_km": 5.2,
  "nodes_covered": 45,
  "pois_covered": 2,
  "existing_bikeways_km": 3.5,
  "new_bikeways_km": 1.7
}
```

## Funcionamiento

1. **Descarga del grafo**: Usa `osmnx` para obtener el grafo de calles de la ciudad especificada
2. **Identificación de ciclovías**: Detecta ciclovías existentes mediante atributos OSM (`highway=cycleway`, `cycleway=*`)
3. **Clasificación de aristas**:
   - Ciclovías existentes: marcadas como `fixed=1` (siempre presentes)
   - Candidatas: resto de aristas que pueden convertirse en ciclovías
4. **Cálculo de métricas**:
   - Longitud de cada arista
   - Coordenadas de inicio y fin
   - Pesos adicionales para aristas cerca de POIs
5. **Heurística de selección**: Selecciona nuevas ciclovías priorizando:
   - Aristas que conectan con POIs (mayor peso)
   - Aristas más cortas
   - Respetando el presupuesto máximo
6. **Respuesta**: Retorna la solución con estadísticas y lista de aristas seleccionadas

## Ejemplo de uso con cURL

```bash
curl -X POST "http://localhost:8000/optimize_bikeway" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Piedmont, California, USA",
    "poi": [
      {"lat": 37.8244, "lon": -122.2315, "weight": 10.0},
      {"lat": 37.8254, "lon": -122.2325, "weight": 5.0}
    ],
    "budget_km": 5.0,
    "lambda_cost": 1.0
  }'
```

## Ejemplo de uso con Python

```python
import requests

url = "http://localhost:8000/optimize_bikeway"
data = {
    "city": "Piedmont, California, USA",
    "poi": [
        {"lat": 37.8244, "lon": -122.2315, "weight": 10.0},
        {"lat": 37.8254, "lon": -122.2325, "weight": 5.0}
    ],
    "budget_km": 5.0,
    "lambda_cost": 1.0
}

response = requests.post(url, json=data)
result = response.json()

print(f"Total length: {result['total_length_km']} km")
print(f"New bikeways: {result['new_bikeways_km']} km")
print(f"POIs covered: {result['pois_covered']}")
```

## Estructura

```
qbikes-back/
├── app/
│   ├── api/routes/          # Endpoints
│   ├── services/            # Business logic
│   ├── models/              # Pydantic schemas
│   ├── utils/               # Helper functions
│   └── main.py              # FastAPI app
├── config.py
├── requirements.txt
├── run.py                   # Entry point
└── readme.md
```

## Tecnologías

- **FastAPI** - Framework web moderno
- **OSMnx** - Datos de OpenStreetMap
- **NetworkX** - Análisis de grafos
- **Pydantic** - Validación de datos
- **Qiskit** - Optimización cuántica experimental (Árbol de Steiner)

## 🧠 Ejemplo cuántico: Árbol de Steiner para ciclovías

El repositorio incluye el script `steiner_tree_qaoa.py`, que construye un modelo QUBO del
problema de Árbol de Steiner aplicado a la planificación de ciclovías. El modelo se resuelve
utilizando **Qiskit** y QAOA (Quantum Approximate Optimization Algorithm), y visualiza el
subgrafo óptimo con `networkx`.

### Ejecutar el ejemplo

```powershell
pip install -r requirements.txt
python steiner_tree_qaoa.py
```

El script utiliza los nodos y aristas de ejemplo contenidos en `data.json`, marca los
terminales (universidades, plazas y estaciones) en rojo, los nodos de Steiner utilizados en
azul y las aristas del árbol resultante en verde.
