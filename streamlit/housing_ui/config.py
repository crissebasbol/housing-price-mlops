import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    api_url: str = field(default_factory=lambda: _env("API_URL", "http://api:8000"))
    mlflow_url: str = field(
        default_factory=lambda: _env("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    mlflow_public_url: str = field(
        default_factory=lambda: _env("MLFLOW_PUBLIC_URL", "http://10.43.97.106:8003")
    )
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "housing-price-model"))
    db_host: str = field(default_factory=lambda: _env("DATA_DB_HOST", "postgres-data"))
    db_port: int = field(default_factory=lambda: int(_env("DATA_DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: _env("DATA_DB_NAME", "housing"))
    db_user: str = field(default_factory=lambda: _env("DATA_DB_USER", "app_reader"))
    db_password: str = field(default_factory=lambda: _env("DATA_DB_PASSWORD", ""))
    timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "15")))
    git_sha: str = field(default_factory=lambda: _env("GIT_SHA", "unknown"))


SETTINGS = Settings()
