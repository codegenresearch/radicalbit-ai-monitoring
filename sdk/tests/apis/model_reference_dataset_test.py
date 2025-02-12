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
            **{
                "method": responses.GET,
                "url": f"{base_url}/api/models/{str(model_id)}/reference/statistics",
                "status": 200,
                "body": f"""{{\n                    "datetime": "something_not_used",\n                    "jobStatus": "SUCCEEDED",\n                    "statistics": {{\n                        "nVariables": {n_variables},\n                        "nObservations": {n_observations},\n                        "missingCells": {missing_cells},\n                        "missingCellsPerc": {missing_cells_perc},\n                        "duplicateRows": {duplicate_rows},\n                        "duplicateRowsPerc": {duplicate_rows_perc},\n                        "numeric": {numeric},\n                        "categorical": {categorical},\n                        "datetime": {datetime}\n                    }}\n                }}""",
            }
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
            **{
                "method": responses.GET,
                "url": f"{base_url}/api/models/{str(model_id)}/reference/statistics",
                "status": 200,
                "body": '{"statistics": "wrong"}',
            }
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
            **{
                "method": responses.GET,
                "url": f"{base_url}/api/models/{str(model_id)}/reference/statistics",
                "status": 200,
                "body": '{"wrong": "json"}',
            }
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
            **{
                "method": responses.GET,
                "url": f"{base_url}/api/models/{str(model_id)}/reference/model-quality",
                "status": 200,
                "body": f"""{{\n                    "datetime": "something_not_used",\n                    "jobStatus": "SUCCEEDED",\n                    "modelQuality": {{\n                        "f1": {f1},\n                        "accuracy": {accuracy},\n                        "precision": {precision},\n                        "recall": {recall},\n                        "fMeasure": {f_measure},\n                        "weightedPrecision": {weighted_precision},\n                        "weightedRecall": {weighted_recall},\n                        "weightedFMeasure": {weighted_f_measure},\n                        "weightedTruePositiveRate": {weighted_true_positive_rate},\n                        "weightedFalsePositiveRate": {weighted_false_positive_rate},\n                        "truePositiveRate": {true_positive_rate},\n                        "falsePositiveRate": {false_positive_rate},\n                        "areaUnderRoc": {area_under_roc},\n                        "areaUnderPr": {area_under_pr},\n                        "truePositiveCount": {true_positive_count},\n                        "falsePositiveCount": {false_positive_count},\n                        "trueNegativeCount": {true_negative_count},\n                        "falseNegativeCount": {false_negative_count}\n                    }}\n                }}""",
            }
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
            **{
                "method": responses.GET,
                "url": f"{base_url}/api/models/{str(model_id)}/reference/model-quality",
                "status": 200,
                "body": '{"modelQuality": "wrong"}',
            }
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
            **{
                "method": responses.GET,
                "url": f"{base_url}/api/models/{str(model_id)}/reference/model-quality",
                "status": 200,
                "body": '{"wrong": "json"}',
            }
        )

        with self.assertRaises(ClientError):
            model_reference_dataset.model_quality()