# QBikes - QAOA Steiner Tree para Planificación de Ciclovías

Implementación de algoritmo QAOA (Quantum Approximate Optimization Algorithm) para resolver el problema del Árbol de Steiner aplicado a la planificación óptima de redes de ciclovías usando datos reales de OpenStreetMap.

## 🎯 Características

- 🚴 **Datos Reales**: Obtención de grafos de calles y ciclovías existentes desde OpenStreetMap usando `osmnx`
- 🔬 **QAOA**: Implementación completa del algoritmo cuántico QAOA con Qiskit
- 🌳 **Steiner Tree**: Formulación QUBO del problema de Árbol de Steiner
- 🎨 **Visualización**: Mapas interactivos con Folium y gráficos con Matplotlib
- 📊 **Métricas**: Análisis completo de conectividad y calidad de soluciones
- 📓 **Jupyter Notebook**: Implementación educativa paso a paso

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Abrir el notebook
jupyter notebook steiner_tree_demo.ipynb
```

El notebook incluye:
- Formulación matemática del problema QUBO
- Descarga de datos OSM de Montevideo
- Selección inteligente de terminales (ciclovías existentes)
- Optimización QAOA con Qiskit
- Visualizaciones interactivas y estáticas

## 📁 Archivos Principales

- **`steiner_tree_demo.ipynb`** - Notebook principal con implementación completa
- **`steiner_tree_qaoa.py`** - Script Python standalone
- **`qaoa.py`** - Módulo con funciones QAOA reutilizables
- **`QUBO_FORMULATION.md`** - Documentación de la formulación matemática
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

## 🧮 Problema del Árbol de Steiner

El problema consiste en conectar un conjunto de nodos terminales (ciclovías existentes) usando el menor costo posible, pudiendo usar nodos intermedios (Steiner nodes) de la red de calles.

**Formulación QUBO:**
- Variables binarias para nodos y aristas
- 5 penalizaciones diferentes:
  1. Terminal degree (terminales deben tener grado ≥1)
  2. Steiner degree (nodos Steiner usados deben tener grado ≥2)
  3. Connectivity (garantizar árbol conexo)
  4. Tree structure (evitar ciclos, |E| = |V| - 1)
  5. Edge consistency (aristas solo si ambos nodos están seleccionados)

Ver `QUBO_FORMULATION.md` para detalles matemáticos completos.

## 📊 Resultados

**Problema de ejemplo (Montevideo):**
- 3 terminales distribuídos geográficamente
- 9 nodos totales, 8 aristas
- 17 variables QUBO
- Tiempo de ejecución: ~7 minutos
- Solución: 1,619 metros de conexiones
- ✅ Todos los terminales conectados
- ✅ Estructura de árbol válida

## 🛠️ Tecnologías

- **Qiskit 0.45.3** - Framework de computación cuántica
- **qiskit-optimization 0.6.1** - Conversión de problemas a QUBO
- **qiskit-aer 0.13.3** - Simulador cuántico (para problemas grandes)
- **OSMnx 1.9.1** - Datos de OpenStreetMap
- **NetworkX 3.2.1** - Análisis de grafos
- **Folium** - Mapas interactivos
- **Matplotlib** - Visualizaciones estáticas

## 📖 Uso del Script Python

```bash
# Ejecutar script standalone
python steiner_tree_qaoa.py
```

El script carga datos de Montevideo, ejecuta QAOA y genera visualizaciones.

## 📝 Estructura del Proyecto

```
qbikes-back/
├── steiner_tree_demo.ipynb    # Notebook principal (RECOMENDADO)
├── steiner_tree_qaoa.py        # Script standalone
├── qaoa.py                     # Funciones QAOA reutilizables
├── QUBO_FORMULATION.md         # Documentación matemática
├── requirements.txt            # Dependencias
├── steiner_tree_montevideo.html # Mapa interactivo generado
└── cache/                      # Cache de OSM
```

## 🎓 Referencias

- [QAOA Paper (Farhi et al.)](https://arxiv.org/abs/1411.4028)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Steiner Tree Problem](https://en.wikipedia.org/wiki/Steiner_tree_problem)
- [QUBO Formulation](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)

## 📄 Licencia

Este proyecto es código educativo y de investigación. Ver archivos individuales para detalles de licencia.
