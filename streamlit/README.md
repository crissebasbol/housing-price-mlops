# Interfaz Streamlit

Aplicación para usuarios del modelo. Corre en `nodo-serving` y se publica en `http://10.43.97.106:8007`.

| Página | Qué muestra | De dónde lee |
|---|---|---|
| Predicción | Formulario de la propiedad, precio estimado con versión y alias del modelo, explicación SHAP en cascada ("¿Por qué este precio?") y envío del precio real | API (`/predict`, `/explain`, `/feedback`) con `X-Traffic-Source: streamlit` |
| Historial de lotes | Cada ingreso: llegada, filas válidas, señales y razón de entrenar, MAPE candidata frente a champion, promoción y razón, enlace a MLflow; MAPE del champion por lote (incluye monitoreo de repetidos) y detalle de esquema, calidad, categorías nuevas, drift y compuertas | `audit.batch_runs` con el rol `app_reader` (solo lectura) |
| Modelo en producción | Versión que sirve la API y la que está en sombra, alias `champion`, `challenger` y `previous_champion` con su MAPE de holdout, y las últimas recargas y rollbacks | API (`/model-info`), registro de MLflow y `audit.model_actions` |

La interfaz nunca carga el modelo: todas las predicciones pasan por la API, así lo que ve el usuario es exactamente lo que sirve producción y queda registrado para el drift de inferencias.

## Variables

| Variable | Origen | Valor |
|---|---|---|
| `API_URL`, `MLFLOW_TRACKING_URI`, `MLFLOW_PUBLIC_URL`, `MODEL_NAME`, `DATA_DB_*` | ConfigMap `streamlit` | `http://api:8000`, `http://mlflow:5000`, `http://10.43.97.106:8003`, `housing-price-model`, `postgres-data/housing/app_reader` |
| `DATA_DB_PASSWORD` | Secret `postgres-data` | clave `APP_READER_PASSWORD` |

## Desarrollo

```bash
uv run pytest streamlit/tests
docker build -t housing-streamlit:local streamlit
```
