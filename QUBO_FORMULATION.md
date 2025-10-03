# Formulación QUBO para el Árbol de Steiner con QAOA

Este documento explica la formulación matemática del problema de Árbol de Steiner aplicado a la planificación de ciclovías, tal como está implementado en `steiner_tree_qaoa.py`.

## Índice
1. [Introducción](#introducción)
2. [Variables de Decisión](#variables-de-decisión)
3. [Función Objetivo](#función-objetivo)
4. [Penalizaciones](#penalizaciones)
5. [Restricciones Duras](#restricciones-duras)
6. [Hamiltoniano Completo](#hamiltoniano-completo)
7. [Parámetros de Configuración](#parámetros-de-configuración)

---

## Introducción

El problema del **Árbol de Steiner** busca encontrar un subgrafo conexo de costo mínimo que conecte un conjunto de nodos terminales (obligatorios), pudiendo usar nodos de Steiner (opcionales) si reducen el costo total.

En el contexto de ciclovías:
- **Terminales**: puntos clave que deben estar conectados (universidades, estaciones de transporte, plazas)
- **Nodos Steiner**: intersecciones que ya tienen ciclovías existentes
- **Nodos auxiliares**: otras intersecciones que pueden usarse si ayudan
- **Objetivo**: minimizar la longitud total de ciclovías necesarias

---

## Variables de Decisión

### Variables binarias de nodos
Para cada nodo $v \in V$ del grafo:

$$y_v \in \{0, 1\}$$

donde:
- $y_v = 1$ si el nodo $v$ está activo (forma parte de la solución)
- $y_v = 0$ si el nodo $v$ no se usa

### Variables binarias de aristas
Para cada arista $e = (u, v) \in E$ del grafo:

$$x_e = x_{uv} \in \{0, 1\}$$

donde:
- $x_{uv} = 1$ si la arista entre $u$ y $v$ está seleccionada
- $x_{uv} = 0$ si la arista no se usa

---

## Función Objetivo

La función objetivo del QUBO tiene múltiples componentes que se combinan en un Hamiltoniano:

$$H_{\text{total}} = H_{\text{costo}} + H_{\text{nodos}} + H_{\text{consistencia}} + H_{\text{conectividad}} + H_{\text{árbol}}$$

### 1. Costo de las aristas ($H_{\text{costo}}$)

Minimiza la suma de las longitudes de las aristas seleccionadas:

$$H_{\text{costo}} = \alpha \sum_{e \in E} w_e \cdot x_e$$

donde:
- $w_e$ es el peso (longitud en metros) de la arista $e$
- $\alpha$ es el parámetro `cost_scaling` (típicamente 0.0005)

**Implementación:**
```python
def _add_edge_cost_component(self):
    for var_name, cost in self.edge_costs.items():
        self.linear_terms[var_name] += self.config.cost_scaling * cost
```

### 2. Penalización por activación de nodos ($H_{\text{nodos}}$)

Desalienta el uso innecesario de nodos no terminales:

$$H_{\text{nodos}} = \sum_{v \in V \setminus T} p_v \cdot y_v$$

donde:
- $T$ es el conjunto de terminales (siempre activos)
- $p_v = \beta_{\text{Steiner}}$ si $v$ es candidato Steiner (típicamente 0.2)
- $p_v = \beta_{\text{aux}}$ si $v$ es auxiliar (típicamente 0.6)

**Implementación:**
```python
def _add_node_activation_penalties(self):
    for node_id, var_name in self.node_var_names.items():
        if node_id in self.terminals:
            continue  # Terminales no tienen penalización
        penalty = (
            self.config.steiner_activation_penalty
            if node_id in self.steiner_candidates
            else self.config.auxiliary_activation_penalty
        )
        self.linear_terms[var_name] += penalty
```

---

## Penalizaciones

Las penalizaciones se implementan como términos cuadráticos de la forma $(expresión)^2$ para forzar que ciertas condiciones se cumplan.

### 3. Consistencia arista-nodo ($H_{\text{consistencia}}$)

**Regla:** Si una arista está seleccionada, ambos nodos extremos deben estar activos.

Para cada arista $e = (u, v)$:

$$H_{\text{consistencia}} = \lambda_{\text{edge-node}} \sum_{(u,v) \in E} \left[ (x_{uv} - y_u)^2 + (x_{uv} - y_v)^2 \right]$$

Expandiendo $(x_{uv} - y_u)^2$:
$$(x_{uv} - y_u)^2 = x_{uv}^2 - 2x_{uv}y_u + y_u^2 = x_{uv} - 2x_{uv}y_u + y_u$$

(Nota: para variables binarias $x^2 = x$)

Esto genera:
- **Términos lineales:** $+\lambda(x_{uv} + y_u)$
- **Términos cuadráticos:** $-2\lambda \cdot x_{uv}y_u$

**Implementación:**
```python
def _add_edge_node_consistency_penalty(self):
    weight = self.config.lambda_edge_node  # típicamente 45.0
    for (u, v), edge_var in self.edge_var_names.items():
        u_var = self.node_var_names[u]
        v_var = self.node_var_names[v]
        # Penaliza (x_uv - y_u)^2
        self._add_squared_penalty([(edge_var, 1.0), (u_var, -1.0)], weight)
        # Penaliza (x_uv - y_v)^2
        self._add_squared_penalty([(edge_var, 1.0), (v_var, -1.0)], weight)
```

### 4. Conectividad de terminales ($H_{\text{conectividad}}$)

**Regla:** Cada terminal debe tener al menos una arista incidente (grado ≥ 1).

Para cada terminal $t \in T$, sea $\delta(t)$ el conjunto de aristas incidentes:

$$H_{\text{conectividad}} = \lambda_{\text{term-deg}} \sum_{t \in T} \left( d_{\text{target}}(t) - \sum_{e \in \delta(t)} x_e \right)^2$$

donde:
- $d_{\text{target}}(t) = 1$ si $t$ es terminal hoja (primer o último terminal)
- $d_{\text{target}}(t) = 2$ si $t$ es terminal intermedio (fuerza conexión de paso)

Expandiendo:
$$\left( d - \sum_i x_i \right)^2 = d^2 - 2d\sum_i x_i + \left(\sum_i x_i\right)^2$$

Esto genera:
- **Constante:** $+\lambda \cdot d^2$
- **Términos lineales:** $-2\lambda d \cdot x_i$ para cada arista incidente
- **Términos cuadráticos:** $+\lambda \cdot x_i x_j$ para cada par de aristas incidentes
- **Términos diagonales:** $+\lambda \cdot x_i^2 = +\lambda \cdot x_i$

**Implementación:**
```python
def _add_terminal_connectivity_penalty(self):
    weight = self.config.lambda_terminal_deg  # típicamente 650.0
    for terminal in self.terminals:
        incident_edge_vars = [...]  # aristas incidentes al terminal
        
        # Determinar grado objetivo
        if terminal in self.leaf_terminals or len(incident_edge_vars) <= 1:
            target_degree = 1.0
        else:
            target_degree = 2.0
        
        expression = [(var, -1.0) for var in incident_edge_vars]
        self._add_squared_penalty(expression, weight, constant=target_degree)
```

### 5. Balance de árbol ($H_{\text{árbol}}$)

**Regla:** Un árbol conexo con $n$ nodos tiene exactamente $n-1$ aristas.

$$H_{\text{árbol}} = \lambda_{\text{balance}} \left( 1 + \sum_{e \in E} x_e - \sum_{v \in V} y_v \right)^2$$

Esto desalienta ciclos y componentes desconectadas.

Expandiendo:
$$\left(1 + \sum_e x_e - \sum_v y_v\right)^2$$

Genera términos lineales, cuadráticos y constantes que incentivan $|E| = |V| - 1$.

**Implementación:**
```python
def _add_tree_balance_penalty(self):
    weight = self.config.lambda_tree_balance  # típicamente 80.0
    expression = []
    expression.extend([(var, 1.0) for var in self.edge_var_names.values()])
    expression.extend([(var, -1.0) for var in self.node_var_names.values()])
    self._add_squared_penalty(expression, weight, constant=1.0)
```

---

## Restricciones Duras

Además de las penalizaciones suaves, se imponen restricciones lineales estrictas:

### Activación forzada de terminales

Para cada terminal $t \in T$:

$$y_t = 1$$

Esto se implementa como una restricción lineal de igualdad en el `QuadraticProgram`.

**Implementación:**
```python
def _add_terminal_activation_constraints(self):
    for terminal in self.terminals:
        var_name = self.node_var_names[terminal]
        self.qp.linear_constraint(
            linear={var_name: 1.0}, 
            sense="==", 
            rhs=1.0, 
            name=f"terminal_active_{terminal}"
        )
```

---

## Hamiltoniano Completo

El Hamiltoniano total tiene la forma:

$$
\begin{aligned}
H &= \alpha \sum_{e \in E} w_e x_e \\
  &\quad + \sum_{v \in V \setminus T} p_v y_v \\
  &\quad + \lambda_1 \sum_{(u,v) \in E} \left[ (x_{uv} - y_u)^2 + (x_{uv} - y_v)^2 \right] \\
  &\quad + \lambda_2 \sum_{t \in T} \left( d_t - \sum_{e \in \delta(t)} x_e \right)^2 \\
  &\quad + \lambda_3 \left( 1 + \sum_{e \in E} x_e - \sum_{v \in V} y_v \right)^2
\end{aligned}
$$

sujeto a:
$$y_t = 1, \quad \forall t \in T$$

Este Hamiltoniano se traduce a:
- **Términos lineales** sobre variables individuales ($x_e$, $y_v$)
- **Términos cuadráticos** sobre pares de variables ($x_i x_j$, $x_i y_j$, $y_i y_j$)
- **Offset constante** (no afecta la optimización relativa)

---

## Parámetros de Configuración

Los valores por defecto ajustados empíricamente son:

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `lambda_edge_node` | 45.0 | Consistencia arista-nodo |
| `lambda_terminal_deg` | 650.0 | Conectividad de terminales |
| `lambda_tree_balance` | 80.0 | Estructura de árbol |
| `steiner_activation_penalty` | 0.2 | Costo de activar nodo Steiner |
| `auxiliary_activation_penalty` | 0.6 | Costo de activar nodo auxiliar |
| `cost_scaling` | 0.0005 | Escala de costos de aristas |

**Configuración en código:**
```python
SolverConfig(
    terminal_ids=terminals,
    leaf_terminal_ids=leaf_terminals,
    steiner_candidate_ids=sorted(steiner_nodes),
    lambda_edge_node=45.0,
    lambda_terminal_deg=650.0,
    lambda_tree_balance=80.0,
    steiner_activation_penalty=0.2,
    auxiliary_activation_penalty=0.6,
    cost_scaling=0.0005,
)
```

### Ajuste de parámetros

- **Incrementar $\lambda_{\text{term-deg}}$**: fuerza conectividad más estricta de terminales
- **Incrementar $\lambda_{\text{balance}}$**: reduce ciclos, fuerza estructura de árbol
- **Reducir `cost_scaling`**: prioriza cumplir restricciones sobre minimizar longitud
- **Ajustar penalizaciones de nodos**: controla qué tipo de nodos se prefieren

---

## Ejecución con QAOA

El QUBO se resuelve usando el **Quantum Approximate Optimization Algorithm (QAOA)**:

1. **Conversión a Ising:** El `MinimumEigenOptimizer` convierte automáticamente el QUBO a un Hamiltoniano Ising que actúa sobre qubits.

2. **Ansatz parametrizado:** QAOA construye un circuito cuántico con parámetros $(\gamma, \beta)$ que se optimizan clásicamente.

3. **Optimización clásica:** COBYLA ajusta los parámetros para minimizar el valor esperado del Hamiltoniano.

4. **Muestreo:** El sampler (Statevector, Aer o clásico) ejecuta el circuito y mide las variables.

5. **Decodificación:** La solución binaria se mapea de vuelta a nodos y aristas seleccionadas.

**Configuración QAOA:**
```python
qaoa = QAOA(
    sampler=sampler,           # StatevectorSampler o AerSampler
    optimizer=COBYLA(maxiter=150),
    reps=2,                    # Profundidad del ansatz (p=2)
    initial_point=np.zeros(4)  # 2*reps parámetros
)
```

---

## Referencias

- **QAOA:** Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm"
- **QUBO:** Glover, F., et al. (2019). "A Tutorial on Formulating and Using QUBO Models"
- **Steiner Tree:** Hwang, F., Richards, D., & Winter, P. (1992). "The Steiner Tree Problem"

---

## Ver también

- [steiner_tree_qaoa.py](steiner_tree_qaoa.py) - Implementación completa
- [README.md](readme.md) - Guía de uso
- [requirements.txt](requirements.txt) - Dependencias (Qiskit)
