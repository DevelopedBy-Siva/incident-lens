import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from app.data.ingestion.datadog_connector import (
    DatadogLogConnector,
    DatadogLogSourceConfig,
)
from app.data.ingestion.log_source import LogSourceAuthenticationError
from app.data.ingestion.log_source_watcher import (
    LOOKBACK_SECONDS,
    poll_project,
    process_envelopes,
)


def _project(**overrides):
    values = {
        "id": "project-1",
        "name": "alpha",
        "datadog_api_key": "api-key",
        "datadog_app_key": "app-key",
        "datadog_site": "datadoghq.com",
        "datadog_query": "status:error",
        "datadog_service": "checkout",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DatadogConnectorTests(unittest.TestCase):
    def test_project_configuration_does_not_use_deployment_fallbacks(self):
        project = _project(datadog_api_key=None)
        with self.assertRaisesRegex(ValueError, "API key"):
            DatadogLogSourceConfig.from_project(project)

    def test_search_authenticates_pages_and_normalizes_in_timestamp_order(self):
        requests = []

        def handler(request):
            requests.append(request)
            body = json.loads(request.content)
            if "cursor" not in body["page"]:
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "data": [
                            {
                                "id": "later",
                                "attributes": {
                                    "message": "ERROR later",
                                    "timestamp": "2026-01-01T00:00:02Z",
                                    "service": "payments",
                                    "tags": ["env:staging"],
                                    "attributes": {"trace_id": "trace-2"},
                                },
                            }
                        ],
                        "meta": {"page": {"after": "next-page"}},
                    },
                )
            return httpx.Response(
                200,
                request=request,
                json={
                    "data": [
                        {
                            "id": "earlier",
                            "attributes": {
                                "message": "WARN earlier",
                                "timestamp": "2026-01-01T00:00:01Z",
                                "service": "checkout",
                                "tags": ["env:prod"],
                            },
                        }
                    ],
                    "meta": {"page": {}},
                },
            )

        config = DatadogLogSourceConfig.from_project(_project())
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            connector = DatadogLogConnector(config, client=client, page_size=25)
            envelopes = connector.fetch(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual([item.provider_id for item in envelopes], ["earlier", "later"])
        self.assertEqual(envelopes[0].source, "checkout")
        self.assertEqual(envelopes[0].environment, "prod")
        self.assertEqual(envelopes[1].attributes, {"trace_id": "trace-2"})
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].headers["DD-API-KEY"], "api-key")
        self.assertEqual(requests[0].headers["DD-APPLICATION-KEY"], "app-key")
        first_body = json.loads(requests[0].content)
        self.assertEqual(first_body["page"]["limit"], 25)
        self.assertEqual(
            first_body["filter"]["query"],
            '(status:error) AND env:"prod" AND service:"checkout"',
        )
        self.assertEqual(json.loads(requests[1].content)["page"]["cursor"], "next-page")

    def test_authentication_failure_has_a_specific_error(self):
        def handler(request):
            return httpx.Response(403, request=request, text="forbidden")

        config = DatadogLogSourceConfig.from_project(_project())
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            connector = DatadogLogConnector(config, client=client)
            with self.assertRaises(LogSourceAuthenticationError):
                connector.fetch(
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                )

    def test_project_configuration_is_isolated(self):
        captured = []

        def handler(request):
            captured.append(
                (
                    request.headers["DD-API-KEY"],
                    request.headers["DD-APPLICATION-KEY"],
                    json.loads(request.content)["filter"]["query"],
                )
            )
            return httpx.Response(200, request=request, json={"data": []})

        projects = [
            _project(),
            _project(
                id="project-2",
                datadog_api_key="api-key-2",
                datadog_app_key="app-key-2",
                datadog_service="billing",
            ),
        ]
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            for project in projects:
                DatadogLogConnector(
                    DatadogLogSourceConfig.from_project(project), client=client
                ).fetch(
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                )

        self.assertEqual(captured[0][:2], ("api-key", "app-key"))
        self.assertIn('service:"checkout"', captured[0][2])
        self.assertEqual(captured[1][:2], ("api-key-2", "app-key-2"))
        self.assertIn('env:"prod"', captured[1][2])
        self.assertIn('service:"billing"', captured[1][2])

    def test_verify_access_uses_a_single_minimal_search(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, request=request, json={"data": []})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            connector = DatadogLogConnector(
                DatadogLogSourceConfig.from_project(_project()), client=client
            )
            connector.verify_access(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual(len(requests), 1)
        body = json.loads(requests[0].content)
        self.assertEqual(body["page"]["limit"], 1)
        self.assertEqual(
            body["filter"]["query"],
            '(status:error) AND env:"prod" AND service:"checkout"',
        )


class DatadogWatcherTests(unittest.TestCase):
    def test_polling_uses_initial_lookback_then_contiguous_cursor(self):
        ranges = []

        class RecordingConnector:
            def __init__(self, config):
                self.config = config

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def fetch(self, start, end):
                ranges.append((start, end))
                return []

        first_end = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
        _, first_cursor = poll_project(
            _project(), None, first_end, connector_factory=RecordingConnector
        )
        second_end = first_end + timedelta(seconds=30)
        _, second_cursor = poll_project(
            _project(),
            first_cursor,
            second_end,
            connector_factory=RecordingConnector,
        )

        self.assertEqual(
            ranges[0],
            (first_end - timedelta(seconds=LOOKBACK_SECONDS), first_end),
        )
        self.assertEqual(ranges[1], (first_end, second_end))
        self.assertEqual(second_cursor, second_end)

    def test_neutral_envelopes_keep_the_existing_pipeline_contract(self):
        from app.data import pipeline
        from app.data.ingestion.log_source import LogEnvelope

        envelopes = [
            LogEnvelope(
                "1", datetime.now(timezone.utc), "ERROR one", "checkout", "prod"
            ),
            LogEnvelope(
                "2", datetime.now(timezone.utc), "ERROR two", "checkout", "prod"
            ),
            LogEnvelope(
                "3", datetime.now(timezone.utc), "WARN three", "billing", "staging"
            ),
        ]
        with patch.object(
            pipeline,
            "process_log_batch",
            return_value={"incidents_created": 1, "incidents_updated": 0, "failed": 0},
        ) as process_batch:
            result = process_envelopes(_project(), envelopes)

        self.assertEqual(process_batch.call_count, 2)
        batches = [call.args[0] for call in process_batch.call_args_list]
        self.assertEqual(batches[0]["project_id"], "project-1")
        self.assertEqual(batches[0]["source"], "checkout")
        self.assertEqual(batches[0]["environment"], "prod")
        self.assertEqual(batches[0]["logs"], ["ERROR one", "ERROR two"])
        self.assertEqual(result["incidents_created"], 2)


if __name__ == "__main__":
    unittest.main()
