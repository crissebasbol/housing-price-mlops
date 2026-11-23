import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from housing_ml.metrics import ape, price_band  # noqa: E402

log = logging.getLogger(__name__)


def prediction_plots(y_true, y_pred, out_dir: Path, prefix: str) -> list[Path]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=2, alpha=0.3)
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(limits, limits, color="black", linewidth=1)
    ax.set(xscale="log", yscale="log", xlabel="Precio real", ylabel="Precio predicho")
    ax.set_title(f"{prefix}: predicho frente a real")
    paths.append(_save(fig, out_dir / f"{prefix}_predicho_vs_real.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(y_pred / y_true), bins=80)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="log10(predicho / real)", ylabel="Filas")
    ax.set_title(f"{prefix}: distribución del error relativo")
    paths.append(_save(fig, out_dir / f"{prefix}_residuos.png"))

    errors = pd.DataFrame({"band": price_band(y_true), "ape": ape(y_true, y_pred)})
    summary = errors.groupby("band")["ape"].mean().reindex(["<300k", "300k-500k", ">500k"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary.index, summary.fillna(0).to_numpy() * 100)
    ax.set(xlabel="Banda de precio", ylabel="MAPE (%)")
    ax.set_title(f"{prefix}: error por banda de precio")
    paths.append(_save(fig, out_dir / f"{prefix}_mape_por_banda.png"))
    return paths


def shap_plots(model, sample: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict]:
    """SHAP del árbol en escala log del precio. Si falla, no detiene el entrenamiento."""
    try:
        import shap

        steps = model.regressor_.named_steps
        encoded = steps["encode"].transform(steps["features"].transform(sample))
        explanation = shap.TreeExplainer(steps["model"])(encoded)
        paths = []
        for name, draw in (
            ("shap_beeswarm.png", lambda: shap.plots.beeswarm(explanation, show=False)),
            ("shap_importancia.png", lambda: shap.plots.bar(explanation, show=False)),
            (
                "shap_dependencia_house_size.png",
                lambda: shap.plots.scatter(explanation[:, "house_size"], show=False),
            ),
        ):
            plt.figure()
            draw()
            paths.append(_save(plt.gcf(), out_dir / name))
        importance = dict(
            zip(encoded.columns, np.abs(explanation.values).mean(axis=0).tolist(), strict=True)
        )
        return paths, importance
    except Exception:
        log.warning("No se pudieron generar las explicaciones SHAP", exc_info=True)
        return [], {}


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
