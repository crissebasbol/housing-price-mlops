import gzip
import io
import json

import boto3
import pandas as pd
from botocore.config import Config

from pipeline.config import DATASETS_BUCKET, LANDING_BUCKET, S3_ENDPOINT_URL


def work_key(ingestion_id: int, name: str) -> str:
    return f"work/ingestion={ingestion_id}/{name}"


def split_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


class Storage:
    """Landing de las respuestas de la API (bucket landing) y archivos de trabajo (bucket datasets).
    Las credenciales vienen de AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY (Secret minio-airflow)."""

    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            region_name="us-east-1",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    def check(self) -> None:
        for bucket in (LANDING_BUCKET, DATASETS_BUCKET):
            self.s3.head_bucket(Bucket=bucket)

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def put_json_gz(self, bucket: str, key: str, payload) -> str:
        body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/gzip")
        return f"s3://{bucket}/{key}"

    def get_json_gz(self, uri: str):
        bucket, key = split_uri(uri)
        body = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(gzip.decompress(body))

    def put_json(self, bucket: str, key: str, payload) -> str:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        return f"s3://{bucket}/{key}"

    def put_parquet(self, key: str, frame: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self.s3.put_object(Bucket=DATASETS_BUCKET, Key=key, Body=buffer.getvalue())

    def get_parquet(self, key: str) -> pd.DataFrame:
        body = self.s3.get_object(Bucket=DATASETS_BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
