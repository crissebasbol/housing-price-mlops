# Housing Price MLOps

**Continuous training MLOps platform for housing price prediction:** Airflow decides when to retrain, MLflow champion/challenger promotion, FastAPI and Streamlit serving, deployed on Kubernetes with GitHub Actions and Argo CD.

Every data batch is validated (schema, quality, drift, new categories) before the pipeline decides, with explicit rules, whether a new model is worth training. Candidates are compared against the production model on a fixed holdout through statistical and per-segment gates, and promotion is a change of alias in the MLflow registry that the inference API picks up without being redeployed. The whole system runs on Kubernetes and is synchronized from this repository by Argo CD.

`mlops` `continuous-training` `airflow` `mlflow` `kubernetes` `argocd` `gitops` `fastapi` `streamlit` `prometheus` `grafana` `locust`

---

Proyecto final de **Operaciones de Machine Learning**, Pontificia Universidad Javeriana (Nivel 4: automatización, decisión de reentrenamiento y despliegue GitOps). Este README es el informe del proyecto y la bitácora de su desarrollo.

**Video de sustentación:** _pendiente_

## Contenido

1. [Arquitectura](#1-arquitectura)
2. [Bitácora](#2-bitácora)
3. [Datos](#3-datos)
4. [Validación y decisión de entrenamiento](#4-validación-y-decisión-de-entrenamiento)
5. [Modelo, métrica y promoción](#5-modelo-métrica-y-promoción)
6. [Calibración](#6-calibración)
7. [Análisis por lote](#7-análisis-por-lote)
8. [MLflow](#8-mlflow)
9. [Inferencia](#9-inferencia)
10. [Streamlit](#10-streamlit)
11. [CI/CD y GitOps](#11-cicd-y-gitops)
12. [Kubernetes](#12-kubernetes)
13. [Observabilidad y prueba de carga](#13-observabilidad-y-prueba-de-carga)
14. [Lo que no se hizo y por qué](#14-lo-que-no-se-hizo-y-por-qué)
15. [Cómo reproducir](#15-cómo-reproducir)
16. [Equipo](#16-equipo)

## 1. Arquitectura

_En construcción: diagrama propio y comunicación entre componentes._

| Componente | Rol |
|---|---|
| API de datos | Imagen del curso desplegada en el clúster; entrega un lote por petición |
| Airflow | Orquesta ingesta, validación, decisión de entrenamiento, registro y promoción |
| PostgreSQL de datos | Schemas `raw`, `clean` y `audit` |
| MLflow + PostgreSQL + MinIO | Experimentos, registro de modelos y artefactos |
| FastAPI | Inferencia con el modelo productivo de MLflow, sin redespliegue al cambiar de versión |
| Streamlit | Predicción e historial de decisiones por lote |
| Prometheus + Grafana | Métricas de la API y del clúster |
| Locust | Pruebas de carga contra la API |
| GitHub Actions + DockerHub | Construcción y publicación de imágenes |
| Argo CD | Sincronización de los manifiestos del repositorio con el clúster |

## 2. Bitácora

### 17 de noviembre: punto de partida

Leímos el enunciado y revisamos lo que ya teníamos de proyectos anteriores. Del proyecto 2 reutilizamos la infraestructura: un clúster RKE2 administrado con Rancher, con un nodo para datos y entrenamiento y otro para inferencia y observabilidad. También la base de Airflow con Helm, MLflow con PostgreSQL y MinIO, y la instrumentación de FastAPI. Del taller 6 tomamos el flujo de GitHub Actions que publica en DockerHub y actualiza el tag en Git para que Argo CD sincronice.

Lo nuevo de este nivel es que el sistema debe **decidir**: si un lote justifica entrenar, si el candidato es mejor que el modelo productivo y cuándo cambiar de versión sin tocar la API.

### 17 al 21 de noviembre: un plan completo antes del código

Antes de escribir una línea de código armamos un plan con todo lo que había que construir:

- cada requisito del enunciado asignado a un componente;
- el diseño del DAG y de sus bifurcaciones;
- las tablas de RAW, CLEAN y auditoría;
- las reglas de calidad, de entrenamiento y de promoción con sus umbrales;
- los recursos de cada contenedor;
- las evidencias que debía dejar cada etapa;
- un cronograma con el orden en que se recorta si falta tiempo.

En ese plan estudiamos varias estrategias para decidir si un modelo nuevo es mejor, tomando como referencia cómo lo hace la industria con el esquema *champion/challenger*:

| Estrategia | Decisión | Por qué |
|---|---|---|
| Comparar solo una métrica global | Descartada | Una mejora en promedio puede ocultar caídas en segmentos (por ejemplo, las casas más caras) |
| Holdout fijo que nunca se usa para entrenar | Adoptada | Champion y candidato se comparan siempre sobre los mismos datos, aunque el split de entrenamiento crezca |
| Compuertas: significancia, segmentos, comportamiento, latencia y tamaño | Adoptada | Es la evaluación offline previa al despliegue que usan los equipos de MLOps |
| Explicaciones con SHAP | Adoptada | Como artefacto de cada modelo y para comparar en qué variables se apoya cada versión |
| Modelo en sombra (*shadow*) | Adoptada | El candidato predice en paralelo sin afectar la respuesta, y se mide su latencia y su diferencia con el productivo |
| Rollback por alias | Adoptada | Volver al modelo anterior es un cambio de alias en MLflow, sin redespliegue |
| Endpoint de feedback con el precio real | Adoptada, como complemento | Casi no habrá feedback de usuarios; la etiqueta real llega con los lotes siguientes, que son la fuente principal de evaluación |
| Promoción solo manual por endpoints | Descartada | El enunciado pide que el pipeline promueva por una regla explícita; lo manual queda para operar (rollback, recarga) |
| Canary con Argo Rollouts, pruebas A/B, namespaces de staging | Descartadas | Sin usuarios reales no son significativas, y versionar el modelo en Git contradice que MLflow sea la fuente de verdad |

### 21 de noviembre: exploración de la API de datos y decisiones de diseño

**La API de datos.** Antes de diseñar las tablas levantamos la imagen `cristiandiaz13/mlops-puj:data-api-pf-v2` en local y pedimos los lotes completos del grupo 1. Así no consumimos lotes de la instancia compartida. Leyendo su código confirmamos que:

- avanza un lote por petición, no por tiempo;
- entrega 6 lotes y luego responde HTTP 400 con el mensaje "Ya se recolectó toda la información minima necesaria";
- el reinicio vuelve a entregar exactamente los mismos lotes.

**Los lotes no son aleatorios.** Cada uno es un corte deliberado del dataset:

| Lote | Filas | Qué lo distingue |
|---|---|---|
| 0 | 73.784 | En venta, precio hasta 300 mil USD, casas de 1.500 pies² o más |
| 1 | 94.551 | En venta, precio hasta 300 mil USD, casas de menos de 1.500 pies² |
| 2 | 230.366 | Todas vendidas (`sold`, categoría nueva), misma distribución numérica |
| 3 | 4.055 | Solo Massachusetts, lote pequeño |
| 4 | 320.696 | Precio entre 300 y 500 mil USD, rango nunca visto |
| 5 | 361.457 | Precio desde 500 mil USD, con valores extremos hasta 515 millones |

No hay nulos ni filas duplicadas, pero sí valores absurdos: precios de 1 USD, casas de 1,5 millones de pies², 444 habitaciones. Midiendo el cambio de distribución de cada lote frente a lo acumulado (PSI y Kolmogorov-Smirnov), el lote 1 cambia por completo el tamaño de las casas, el lote 2 no cambia nada numérico pero trae un estado nuevo, y los lotes 4 y 5 desplazan por completo el precio.

**Decisiones que salen de esa exploración:**

- **API de datos propia dentro del clúster.** La instancia publicada por el docente es compartida: si otro equipo pide lotes del mismo grupo, avanza nuestro contador. Con una instancia propia la secuencia es reproducible y la API de datos queda, como todos los componentes, en Kubernetes.
- **MAPE como métrica prioritaria.** Con precios entre 10 mil y 20 millones de USD, el MAE en dólares queda dominado por las casas más caras y el RMSE por unos pocos extremos. El error porcentual se puede comparar entre una casa de 100 mil y una de 5 millones. MAE y R² quedan como salvaguardas.
- **Reglas de calidad por fila** (por ejemplo, precio entre 10 mil y 20 millones de USD, 1 a 20 habitaciones). Las filas inválidas quedan en RAW con su motivo y no pasan a CLEAN.
- **Fin de datos: reiniciar y seguir leyendo.** Cuando la API se agota, el pipeline la reinicia. Los lotes repetidos se reconocen por su huella SHA-256 y no se reingieren. Se usan para vigilar al modelo productivo sobre filas que nunca vio.
- **Promoción por compuertas, no solo por una métrica.** Un candidato reemplaza al productivo solo si mejora el MAPE en al menos 1 % sobre un holdout fijo que ningún modelo usa para entrenar, si la mejora es estadísticamente significativa, si no empeora en segmentos importantes (banda de precio, estado, `status`) y si cumple un presupuesto de latencia y tamaño.

## 3. Datos

_En construcción: RAW, CLEAN, auditoría y trazabilidad por lote._

## 4. Validación y decisión de entrenamiento

_En construcción._

## 5. Modelo, métrica y promoción

_En construcción._

## 6. Calibración

Antes de llevar las reglas a Airflow corrimos el pipeline fuera del clúster con `scripts/simulate_batches.py`. Aplica la misma lógica del paquete `housing_ml` (esquema, calidad, drift, decisión de entrenamiento y compuertas), sin Airflow ni MLflow, sobre los 6 lotes descargados de nuestra API de datos. Los recorre dos veces, como cuando el pipeline reinicia la API al agotarse. Sirve para calibrar los umbrales y para medir el tiempo y la memoria del entrenamiento.

### Corrida del 21 de noviembre

WSL2 (Ubuntu) en un portátil con Intel Core i7-1355U y 32 GB de RAM; CPython 3.12.16 y scikit-learn 1.8.0.

```bash
uv run python scripts/simulate_batches.py --batches-dir data/batches --out data/simulation
```

| Ciclo | Lote | Filas | Válidas | Entrenamiento | Señales | Promoción | MAPE candidato | MAPE productivo | Entrenamiento (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 73784 | 73635 | baseline | baseline | rejected | 23,26 % | - | 2.4 |
| 1 | 1 | 94551 | 94421 | baseline | baseline | rejected | 24,66 % | - | 10.2 |
| 1 | 2 | 230366 | 230029 | baseline | baseline | rejected | 24,21 % | - | 15.3 |
| 1 | 3 | 4055 | 4054 | small_batch | - | - | - | - | - |
| 1 | 4 | 320696 | 320538 | baseline | baseline | rejected | 20,43 % | - | 32.7 |
| 1 | 5 | 361457 | 360660 | baseline | baseline | rejected | 20,84 % | - | 74.2 |
| 2 | 0 | 73784 | - | - | - | - | - | - | - |
| 2 | 1 | 94551 | - | - | - | - | - | - | - |
| 2 | 2 | 230366 | - | - | - | - | - | - | - |
| 2 | 3 | 4055 | - | - | - | - | - | - | - |
| 2 | 4 | 320696 | - | - | - | - | - | - | - |
| 2 | 5 | 361457 | - | - | - | - | - | - | - |

El MAPE candidato se mide sobre el holdout fijo; el log muestra además el MAPE de validación. La memoria pico del proceso llegó a 2.107 MB al entrenar con el lote 5.

<details>
<summary>Log de la corrida</summary>

```text
=== ciclo 1, lote 0: 73784 filas
Esquema ok; válidas 73635, rechazadas 149 {'price_out_of_range': 65, 'house_size_out_of_range': 25, 'acre_lot_out_of_range': 64, 'bed_out_of_range': 1, 'bath_out_of_range': 1}
PSI: {}
Nuevas: {}
Entrena: línea base, no hay modelo productivo.
Entrenado en 2 s con 51397 filas, 102 iteraciones; val MAPE 23,43 %
  OK  G5 comportamiento: 0,00 % fuera de rango plausible, 0 predicciones bajan al aumentar house_size 20 %, categorías desconocidas sin error
  NO  G6 presupuesto: p95 de una predicción 60,5 ms (máximo 50), modelo de 2,3 MB (máximo 200)
  diagnóstico price_band=<300k: 23,39 % frente a -
Se rechaza: G6 presupuesto: p95 de una predicción 60,5 ms (máximo 50), modelo de 2,3 MB (máximo 200).

=== ciclo 1, lote 1: 94551 filas
Esquema ok; válidas 94421, rechazadas 130 {'price_out_of_range': 71, 'acre_lot_out_of_range': 57, 'bed_out_of_range': 2}
PSI: {'price': 0.3382, 'house_size': 8.3199, 'bed': 1.1151, 'bath': 1.0619, 'acre_lot': 0.1294}
Nuevas: {'status': 0.0, 'state': 0.0, 'city': 0.0521, 'zip_code': 0.102, 'brokered_by': 0.2147}
Entrena: línea base, no hay modelo productivo.
Entrenado en 10 s con 117309 filas, 265 iteraciones; val MAPE 24,59 %
  OK  G5 comportamiento: 0,00 % fuera de rango plausible, 0 predicciones bajan al aumentar house_size 20 %, categorías desconocidas sin error
  NO  G6 presupuesto: p95 de una predicción 81,1 ms (máximo 50), modelo de 5,0 MB (máximo 200)
  diagnóstico price_band=<300k: 24,75 % frente a -
Se rechaza: G6 presupuesto: p95 de una predicción 81,1 ms (máximo 50), modelo de 5,0 MB (máximo 200).

=== ciclo 1, lote 2: 230366 filas
Esquema ok; válidas 230029, rechazadas 337 {'price_out_of_range': 159, 'house_size_out_of_range': 8, 'acre_lot_out_of_range': 168, 'bed_out_of_range': 1, 'bath_out_of_range': 2}
PSI: {'price': 0.0006, 'house_size': 0.0016, 'bed': 0.0023, 'bath': 0.0013, 'acre_lot': 0.0044}
Nuevas: {'status': 1.0, 'state': 0.0, 'city': 0.0235, 'zip_code': 0.0412, 'brokered_by': 0.1016}
Entrena: línea base, no hay modelo productivo.
Entrenado en 15 s con 277963 filas, 311 iteraciones; val MAPE 24,08 %
  OK  G5 comportamiento: 0,00 % fuera de rango plausible, 0 predicciones bajan al aumentar house_size 20 %, categorías desconocidas sin error
  NO  G6 presupuesto: p95 de una predicción 135,0 ms (máximo 50), modelo de 6,0 MB (máximo 200)
  diagnóstico price_band=<300k: 24,33 % frente a -
Se rechaza: G6 presupuesto: p95 de una predicción 135,0 ms (máximo 50), modelo de 6,0 MB (máximo 200).

=== ciclo 1, lote 3: 4055 filas
Esquema ok; válidas 4054, rechazadas 1 {'acre_lot_out_of_range': 1}
PSI: {'price': 0.4081, 'house_size': 0.0669, 'bed': 0.0875, 'bath': 0.1857, 'acre_lot': 0.1094}
Nuevas: {'status': 0.0, 'state': 1.0, 'city': 0.2706, 'zip_code': 1.0, 'brokered_by': 0.6332}
No entrena: lote pequeño (4.054 filas válidas, 1,0 % del acumulado; mínimo 10.000 filas y 5 % del acumulado). Trae categorías nuevas en state (100,0 % del lote), city (27,1 % del lote), zip_code (100,0 % del lote), brokered_by (63,3 % del lote); sus filas quedan en CLEAN para el próximo entrenamiento.

=== ciclo 1, lote 4: 320696 filas
Esquema ok; válidas 320538, rechazadas 158 {'house_size_out_of_range': 5, 'acre_lot_out_of_range': 140, 'bed_out_of_range': 10, 'bath_out_of_range': 6}
PSI: {'price': 8.0748, 'house_size': 0.6023, 'bed': 0.2315, 'bath': 0.5862, 'acre_lot': 0.0109}
Nuevas: {'status': 0.0, 'state': 0.0, 'city': 0.0217, 'zip_code': 0.0563, 'brokered_by': 0.1232}
Entrena: línea base, no hay modelo productivo.
Entrenado en 33 s con 505227 filas, 734 iteraciones; val MAPE 20,22 %
  OK  G5 comportamiento: 0,00 % fuera de rango plausible, 0 predicciones bajan al aumentar house_size 20 %, categorías desconocidas sin error
  NO  G6 presupuesto: p95 de una predicción 259,4 ms (máximo 50), modelo de 12,5 MB (máximo 200)
  diagnóstico price_band=<300k: 27,14 % frente a -
  diagnóstico price_band=300k-500k: 12,20 % frente a -
Se rechaza: G6 presupuesto: p95 de una predicción 259,4 ms (máximo 50), modelo de 12,5 MB (máximo 200).

=== ciclo 1, lote 5: 361457 filas
Esquema ok; válidas 360660, rechazadas 797 {'price_out_of_range': 390, 'house_size_out_of_range': 58, 'acre_lot_out_of_range': 150, 'bed_out_of_range': 217, 'bath_out_of_range': 152}
PSI: {'price': 8.1982, 'house_size': 0.9056, 'bed': 0.3774, 'bath': 0.7372, 'acre_lot': 0.0412}
Nuevas: {'status': 0.0, 'state': 0.0, 'city': 0.0389, 'zip_code': 0.0959, 'brokered_by': 0.1128}
Entrena: línea base, no hay modelo productivo.
Entrenado en 74 s con 757350 filas, 1024 iteraciones; val MAPE 20,80 %
  OK  G5 comportamiento: 0,00 % fuera de rango plausible, 0 predicciones bajan al aumentar house_size 20 %, categorías desconocidas sin error
  NO  G6 presupuesto: p95 de una predicción 328,6 ms (máximo 50), modelo de 17,0 MB (máximo 200)
  diagnóstico price_band=<300k: 28,58 % frente a -
  diagnóstico price_band=>500k: 17,59 % frente a -
  diagnóstico price_band=300k-500k: 14,94 % frente a -
Se rechaza: G6 presupuesto: p95 de una predicción 328,6 ms (máximo 50), modelo de 17,0 MB (máximo 200).

=== ciclo 2, lote 0: 73784 filas
Duplicado del lote 0; MAPE del champion en su holdout: None

=== ciclo 2, lote 1: 94551 filas
Duplicado del lote 1; MAPE del champion en su holdout: None

=== ciclo 2, lote 2: 230366 filas
Duplicado del lote 2; MAPE del champion en su holdout: None

=== ciclo 2, lote 3: 4055 filas
Duplicado del lote 3; MAPE del champion en su holdout: None

=== ciclo 2, lote 4: 320696 filas
Duplicado del lote 4; MAPE del champion en su holdout: None

=== ciclo 2, lote 5: 361457 filas
Duplicado del lote 5; MAPE del champion en su holdout: None
```

</details>

**Lo que muestra esta corrida:**

- **G6 rechazó todos los candidatos por latencia.** El p95 de una predicción subió de 60,5 ms con 102 iteraciones a 328,6 ms con 1.024, siempre por encima del máximo de 50 ms. El tamaño (2,3 a 17,0 MB) quedó lejos del máximo de 200 MB.
- **Nunca hubo modelo productivo.** Como ninguna línea base pasó, las compuertas que comparan contra el productivo (G1 a G4) no llegaron a evaluarse. En el segundo ciclo los lotes repetidos se reconocieron como duplicados, pero no hubo modelo que vigilar.
- **La regla de lote pequeño se comportó como se diseñó.** El lote 3 (4.054 filas válidas, 1,0 % del acumulado) no entrena y sus filas quedan para el siguiente entrenamiento.

## 7. Análisis por lote

_En construcción._

## 8. MLflow

_En construcción._

## 9. Inferencia

_En construcción._

## 10. Streamlit

_En construcción._

## 11. CI/CD y GitOps

_En construcción._

## 12. Kubernetes

_En construcción._

## 13. Observabilidad y prueba de carga

_En construcción._

## 14. Lo que no se hizo y por qué

_En construcción._

## 15. Cómo reproducir

_En construcción._

## 16. Equipo

| Integrante | Responsabilidad |
|---|---|
| Cristhian Bolaños | Pipeline de datos y entrenamiento, API, Streamlit, CI |
| Cristian Organista | Clúster, Argo CD, despliegue y operación |
| Fabián Ramírez | Observabilidad, pruebas de carga y video |

Licencia [MIT](LICENSE).
