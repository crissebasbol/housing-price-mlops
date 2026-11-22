## Qué cambia

<!-- Componente y resumen en 2 o 3 líneas. -->

## Por qué

<!-- Requisito del enunciado, decisión técnica o problema que resuelve. -->

## Cómo se probó

- [ ] `uv run ruff check .` y `uv run ruff format --check .`
- [ ] `uv run pytest`
- [ ] Probado en el clúster (si aplica): <!-- comando o captura -->

## Checklist

- [ ] Sin secretos, `.env`, kubeconfig, modelos ni datos en el diff
- [ ] Si cambian dependencias de ML: mismas versiones de scikit-learn y numpy en Airflow y en la API
- [ ] Contenedores nuevos con `requests`, `limits` y probes
- [ ] README o bitácora actualizados, con capturas si esto genera evidencia
