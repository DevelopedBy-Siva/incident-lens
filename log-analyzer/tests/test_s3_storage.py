import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.training.artifact_metadata import (
    LocalArtifactMetadataWriter,
    S3ArtifactMetadataWriter,
    configured_artifact_metadata_writer,
    configured_artifact_storage_location,
)
from app.training.dataset_storage import (
    LocalDatasetStorage,
    S3DatasetStorage,
    configured_dataset_storage,
    configured_dataset_storage_location,
)


class _Body:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key):
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def get_paginator(self, operation):
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                contents = [
                    {"Key": key}
                    for bucket, key in client.objects
                    if bucket == Bucket and key.startswith(Prefix)
                ]
                return [{"Contents": contents}]

        return Paginator()

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)


class S3StorageTests(unittest.TestCase):
    def test_bucket_configuration_selects_s3_for_datasets_and_artifacts(self):
        client = FakeS3Client()
        environment = {
            "S3_BUCKET": "incident-lens-data",
            "S3_DATASET_PREFIX": "project-datasets",
            "S3_ARTIFACT_PREFIX": "project-adapters",
        }
        with patch.dict(os.environ, environment, clear=False):
            with patch("boto3.client", return_value=client):
                dataset_storage = configured_dataset_storage()
                artifact_storage = configured_artifact_metadata_writer()
            dataset_location = configured_dataset_storage_location()
            artifact_location = configured_artifact_storage_location()

        self.assertIsInstance(dataset_storage, S3DatasetStorage)
        self.assertIsInstance(artifact_storage, S3ArtifactMetadataWriter)
        self.assertEqual(
            dataset_location,
            "s3://incident-lens-data/project-datasets",
        )
        self.assertEqual(
            artifact_location,
            "s3://incident-lens-data/project-adapters",
        )

    def test_missing_bucket_uses_local_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "S3_BUCKET": "",
                "DATASET_STORAGE_PATH": f"{directory}/datasets",
                "ARTIFACT_STORAGE_PATH": f"{directory}/artifacts",
            }
            with patch.dict(os.environ, environment, clear=False):
                dataset_storage = configured_dataset_storage()
                artifact_storage = configured_artifact_metadata_writer()

        self.assertIsInstance(dataset_storage, LocalDatasetStorage)
        self.assertIsInstance(artifact_storage, LocalArtifactMetadataWriter)

    def test_dataset_round_trip(self):
        client = FakeS3Client()
        storage = S3DatasetStorage("bucket", "datasets", client)
        key = storage.storage_key("project-1", "dataset-v1", "jsonl")

        storage.save(key, b'{"value": 1}\n')

        self.assertEqual(key, "datasets/projects/project-1/dataset-v1.jsonl")
        self.assertEqual(storage.load(key), b'{"value": 1}\n')
        storage.delete(key)
        self.assertFalse(client.objects)

    def test_artifact_is_uploaded_and_restored_to_empty_cache(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as first_cache:
            writer = S3ArtifactMetadataWriter(
                first_cache, "bucket", "artifacts", client
            )
            output = Path(writer.prepare("project-1", "adapter-v1"))
            (output / "adapter_model.safetensors").write_bytes(b"weights")
            writer.write("project-1", "adapter-v1", {"artifact_id": "artifact-1"})

        with tempfile.TemporaryDirectory() as second_cache:
            restorer = S3ArtifactMetadataWriter(
                second_cache, "bucket", "artifacts", client
            )
            restored = Path(
                restorer.ensure_local("project-1", "adapter-v1", "/missing/old/cache")
            )
            self.assertEqual(
                json.loads((restored / "metadata.json").read_text()),
                {"artifact_id": "artifact-1"},
            )
            self.assertEqual(
                (restored / "adapter_model.safetensors").read_bytes(), b"weights"
            )


if __name__ == "__main__":
    unittest.main()
