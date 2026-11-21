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
6. [Análisis por lote](#6-análisis-por-lote)
7. [MLflow](#7-mlflow)
8. [Inferencia](#8-inferencia)
9. [Streamlit](#9-streamlit)
10. [CI/CD y GitOps](#10-cicd-y-gitops)
11. [Kubernetes](#11-kubernetes)
12. [Observabilidad y prueba de carga](#12-observabilidad-y-prueba-de-carga)
13. [Lo que no se hizo y por qué](#13-lo-que-no-se-hizo-y-por-qué)
14. [Cómo reproducir](#14-cómo-reproducir)
15. [Equipo](#15-equipo)

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

## 6. Análisis por lote

_En construcción._

## 7. MLflow

_En construcción._

## 8. Inferencia

_En construcción._

## 9. Streamlit

_En construcción._

## 10. CI/CD y GitOps

_En construcción._

## 11. Kubernetes

_En construcción._

## 12. Observabilidad y prueba de carga

_En construcción._

## 13. Lo que no se hizo y por qué

_En construcción._

## 14. Cómo reproducir

_En construcción._

## 15. Equipo

| Integrante | Responsabilidad |
|---|---|
| Cristhian Bolaños | Pipeline de datos y entrenamiento, API, Streamlit, CI |
| Cristian Organista | Clúster, Argo CD, despliegue y operación |
| Fabián Ramírez | Observabilidad, pruebas de carga y video |

Licencia [MIT](LICENSE).
