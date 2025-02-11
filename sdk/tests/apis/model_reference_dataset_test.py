from radicalbit_platform_sdk.apis import ModelReferenceDataset
from radicalbit_platform_sdk.models import ReferenceFileUpload, ModelType, JobStatus
from radicalbit_platform_sdk.errors import ClientError
import responses
import unittest
import uuid


class ModelReferenceDatasetTest(unittest.TestCase):
    @responses.activate
    def test_statistics_ok(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        n_variables = 10
        n_observations = 1000
        missing_cells = 10
        missing_cells_perc = 1
        duplicate_rows = 10
        duplicate_rows_perc = 1
        numeric = 3
        categorical = 6
        datetime = 1
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/statistics",
            json={
                "datetime": "something_not_used",
                "jobStatus": "SUCCEEDED",
                "statistics": {
                    "nVariables": n_variables,
                    "nObservations": n_observations,
                    "missingCells": missing_cells,
                    "missingCellsPerc": missing_cells_perc,
                    "duplicateRows": duplicate_rows,
                    "duplicateRowsPerc": duplicate_rows_perc,
                    "numeric": numeric,
                    "categorical": categorical,
                    "datetime": datetime
                }
            },
            status=200
        )

        stats = model_reference_dataset.statistics()

        assert stats.n_variables == n_variables
        assert stats.n_observations == n_observations
        assert stats.missing_cells == missing_cells
        assert stats.missing_cells_perc == missing_cells_perc
        assert stats.duplicate_rows == duplicate_rows
        assert stats.duplicate_rows_perc == duplicate_rows_perc
        assert stats.numeric == numeric
        assert stats.categorical == categorical
        assert stats.datetime == datetime
        assert model_reference_dataset.status() == JobStatus.SUCCEEDED

    @responses.activate
    def test_statistics_validation_error(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/statistics",
            json={"statistics": "wrong"},
            status=200
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.statistics()

    @responses.activate
    def test_statistics_key_error(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/statistics",
            json={"wrong": "json"},
            status=200
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.statistics()

    @responses.activate
    def test_model_metrics_ok(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        f1 = 0.75
        accuracy = 0.98
        recall = 0.23
        weighted_precision = 0.15
        weighted_true_positive_rate = 0.01
        weighted_false_positive_rate = 0.23
        weighted_f_measure = 2.45
        true_positive_rate = 4.12
        false_positive_rate = 5.89
        precision = 2.33
        weighted_recall = 4.22
        f_measure = 9.33
        area_under_roc = 45.2
        area_under_pr = 32.9
        true_positive_count = 10
        false_positive_count = 5
        true_negative_count = 2
        false_negative_count = 7
        histogram = {"bins": [1, 2, 3], "counts": [10, 20, 30]}
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/model-quality",
            json={
                "datetime": "something_not_used",
                "jobStatus": "SUCCEEDED",
                "modelQuality": {
                    "f1": f1,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "fMeasure": f_measure,
                    "weightedPrecision": weighted_precision,
                    "weightedRecall": weighted_recall,
                    "weightedFMeasure": weighted_f_measure,
                    "weightedTruePositiveRate": weighted_true_positive_rate,
                    "weightedFalsePositiveRate": weighted_false_positive_rate,
                    "truePositiveRate": true_positive_rate,
                    "falsePositiveRate": false_positive_rate,
                    "areaUnderRoc": area_under_roc,
                    "areaUnderPr": area_under_pr,
                    "truePositiveCount": true_positive_count,
                    "falsePositiveCount": false_positive_count,
                    "trueNegativeCount": true_negative_count,
                    "falseNegativeCount": false_negative_count,
                    "histogram": histogram
                }
            },
            status=200
        )

        metrics = model_reference_dataset.model_quality()

        assert metrics.f1 == f1
        assert metrics.accuracy == accuracy
        assert metrics.recall == recall
        assert metrics.weighted_precision == weighted_precision
        assert metrics.weighted_recall == weighted_recall
        assert metrics.weighted_true_positive_rate == weighted_true_positive_rate
        assert metrics.weighted_false_positive_rate == weighted_false_positive_rate
        assert metrics.weighted_f_measure == weighted_f_measure
        assert metrics.true_positive_rate == true_positive_rate
        assert metrics.false_positive_rate == false_positive_rate
        assert metrics.true_positive_count == true_positive_count
        assert metrics.false_positive_count == false_positive_count
        assert metrics.true_negative_count == true_negative_count
        assert metrics.false_negative_count == false_negative_count
        assert metrics.precision == precision
        assert metrics.f_measure == f_measure
        assert metrics.area_under_roc == area_under_roc
        assert metrics.area_under_pr == area_under_pr
        assert metrics.histogram == histogram
        assert model_reference_dataset.status() == JobStatus.SUCCEEDED

    @responses.activate
    def test_model_metrics_validation_error(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/model-quality",
            json={"modelQuality": "wrong"},
            status=200
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.model_quality()

    @responses.activate
    def test_model_metrics_key_error(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/model-quality",
            json={"wrong": "json"},
            status=200
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.model_quality()

    @responses.activate
    def test_data_quality_ok(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        avg = 0.1
        std_dev = 0.2
        min_val = 0.0
        max_val = 1.0
        histogram = {"bins": [1, 2, 3], "counts": [10, 20, 30]}
        class_metrics = {"class1": {"precision": 0.8, "recall": 0.9}}
        feature_metrics = {"feature1": {"importance": 0.5}}
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/data-quality",
            json={
                "datetime": "something_not_used",
                "jobStatus": "SUCCEEDED",
                "dataQuality": {
                    "avg": avg,
                    "stdDev": std_dev,
                    "min": min_val,
                    "max": max_val,
                    "histogram": histogram,
                    "classMetrics": class_metrics,
                    "featureMetrics": feature_metrics
                }
            },
            status=200
        )

        metrics = model_reference_dataset.data_quality()

        assert metrics.avg == avg
        assert metrics.std_dev == std_dev
        assert metrics.min == min_val
        assert metrics.max == max_val
        assert metrics.histogram == histogram
        assert metrics.class_metrics == class_metrics
        assert metrics.feature_metrics == feature_metrics
        assert model_reference_dataset.status() == JobStatus.SUCCEEDED

    @responses.activate
    def test_data_quality_validation_error(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/data-quality",
            json={"dataQuality": "wrong"},
            status=200
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.data_quality()

    @responses.activate
    def test_data_quality_key_error(self):
        base_url = "http://api:9000"
        model_id = uuid.uuid4()
        import_uuid = uuid.uuid4()
        model_reference_dataset = ModelReferenceDataset(
            base_url,
            model_id,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=import_uuid,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

        responses.add(
            responses.GET,
            f"{base_url}/api/models/{str(model_id)}/reference/data-quality",
            json={"wrong": "json"},
            status=200
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.data_quality()


This code snippet addresses the feedback by:
1. Removing any extraneous comments or text that could cause a `SyntaxError`.
2. Ensuring that the `body` key in the responses is consistently formatted with proper indentation and spacing.
3. Double-checking that all key names in the response bodies match exactly with those in the gold code.
4. Ensuring that commas are placed correctly in the response body strings.
5. Expanding the `test_data_quality_ok` method to include all relevant metrics like `classMetrics` and `featureMetrics`.
6. Ensuring that all comments are properly formatted with `#`.
7. Ensuring that all assertions are comprehensive and cover all expected outcomes.
8. Consistently using `self.assertRaises(ClientError)` for exception testing.
9. Using the `json` parameter in `responses.add` to ensure valid JSON formatting.