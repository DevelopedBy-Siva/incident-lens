import json
import tempfile
import unittest
from pathlib import Path

from app.training.artifact_metadata import S3ArtifactMetadataWriter
from app.training.dataset_storage import S3DatasetStorage


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
