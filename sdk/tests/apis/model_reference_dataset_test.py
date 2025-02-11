from radicalbit_platform_sdk.apis import ModelReferenceDataset
from radicalbit_platform_sdk.models import ReferenceFileUpload, ModelType, JobStatus
from radicalbit_platform_sdk.errors import ClientError
import responses
import unittest
import uuid


class ModelReferenceDatasetTest(unittest.TestCase):
    BASE_URL = "http://api:9000"
    MODEL_ID = uuid.uuid4()
    IMPORT_UUID = uuid.uuid4()
    N_VARIABLES = 10
    N_OBSERVATIONS = 1000
    MISSING_CELLS = 10
    MISSING_CELLS_PERC = 1
    DUPLICATE_ROWS = 10
    DUPLICATE_ROWS_PERC = 1
    NUMERIC = 3
    CATEGORICAL = 6
    DATETIME = 1
    F1 = 0.75
    ACCURACY = 0.98
    RECALL = 0.23
    WEIGHTED_PRECISION = 0.15
    WEIGHTED_TRUE_POSITIVE_RATE = 0.01
    WEIGHTED_FALSE_POSITIVE_RATE = 0.23
    WEIGHTED_F_MEASURE = 2.45
    TRUE_POSITIVE_RATE = 4.12
    FALSE_POSITIVE_RATE = 5.89
    PRECISION = 2.33
    WEIGHTED_RECALL = 4.22
    F_MEASURE = 9.33
    AREA_UNDER_ROC = 45.2
    AREA_UNDER_PR = 32.9
    TRUE_POSITIVE_COUNT = 10
    FALSE_POSITIVE_COUNT = 5
    TRUE_NEGATIVE_COUNT = 2
    FALSE_NEGATIVE_COUNT = 7
    AVG = 0.1
    CLASS_METRICS = [
        {
            "name": "class1",
            "count": 500,
            "precision": 0.8,
            "recall": 0.9,
            "f1": 0.85
        },
        {
            "name": "class2",
            "count": 500,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65
        }
    ]
    FEATURE_METRICS = [
        {
            "featureName": "feature1",
            "missingValue": {"value": None, "count": 0},
            "medianMetrics": {"value": 0.5, "count": 1000},
            "classMedianMetrics": {"value": 0.5, "count": 1000},
            "histogram": [{"value": 0.1, "count": 100}, {"value": 0.2, "count": 200}],
            "categoryFrequency": {"category1": 500, "category2": 500},
            "distinctValue": [0.1, 0.2, 0.3]
        },
        {
            "featureName": "feature2",
            "missingValue": {"value": None, "count": 0},
            "medianMetrics": {"value": 0.5, "count": 1000},
            "classMedianMetrics": {"value": 0.5, "count": 1000},
            "histogram": [{"value": 0.1, "count": 100}, {"value": 0.2, "count": 200}],
            "categoryFrequency": {"category1": 500, "category2": 500},
            "distinctValue": [0.1, 0.2, 0.3]
        }
    ]

    def setUp(self):
        self.model_reference_dataset = ModelReferenceDataset(
            self.BASE_URL,
            self.MODEL_ID,
            ModelType.BINARY,
            ReferenceFileUpload(
                uuid=self.IMPORT_UUID,
                path="s3://bucket/file.csv",
                date="2014",
                status=JobStatus.IMPORTING,
            ),
        )

    @responses.activate
    def test_statistics_ok(self):
        response_body = f"""{{
            "datetime": "something_not_used",
            "jobStatus": "SUCCEEDED",
            "statistics": {{
                "nVariables": {self.N_VARIABLES},
                "nObservations": {self.N_OBSERVATIONS},
                "missingCells": {self.MISSING_CELLS},
                "missingCellsPerc": {self.MISSING_CELLS_PERC},
                "duplicateRows": {self.DUPLICATE_ROWS},
                "duplicateRowsPerc": {self.DUPLICATE_ROWS_PERC},
                "numeric": {self.NUMERIC},
                "categorical": {self.CATEGORICAL},
                "datetime": {self.DATETIME}
            }}
        }}"""

        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/statistics",
                "status": 200,
                "body": response_body,
                "content_type": "application/json",
            }
        )

        stats = self.model_reference_dataset.statistics()

        assert stats.n_variables == self.N_VARIABLES
        assert stats.n_observations == self.N_OBSERVATIONS
        assert stats.missing_cells == self.MISSING_CELLS
        assert stats.missing_cells_perc == self.MISSING_CELLS_PERC
        assert stats.duplicate_rows == self.DUPLICATE_ROWS
        assert stats.duplicate_rows_perc == self.DUPLICATE_ROWS_PERC
        assert stats.numeric == self.NUMERIC
        assert stats.categorical == self.CATEGORICAL
        assert stats.datetime == self.DATETIME
        assert self.model_reference_dataset.status() == JobStatus.SUCCEEDED

    @responses.activate
    def test_statistics_validation_error(self):
        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/statistics",
                "status": 200,
                "body": '{"statistics": "wrong"}',
            }
        )

        with self.assertRaises(ClientError):
            self.model_reference_dataset.statistics()

    @responses.activate
    def test_statistics_key_error(self):
        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/statistics",
                "status": 200,
                "body": '{"wrong": "json"}',
            }
        )

        with self.assertRaises(ClientError):
            self.model_reference_dataset.statistics()

    @responses.activate
    def test_model_metrics_ok(self):
        response_body = f"""{{
            "datetime": "something_not_used",
            "jobStatus": "SUCCEEDED",
            "modelQuality": {{
                "f1": {self.F1},
                "accuracy": {self.ACCURACY},
                "precision": {self.PRECISION},
                "recall": {self.RECALL},
                "fMeasure": {self.F_MEASURE},
                "weightedPrecision": {self.WEIGHTED_PRECISION},
                "weightedRecall": {self.WEIGHTED_RECALL},
                "weightedFMeasure": {self.WEIGHTED_F_MEASURE},
                "weightedTruePositiveRate": {self.WEIGHTED_TRUE_POSITIVE_RATE},
                "weightedFalsePositiveRate": {self.WEIGHTED_FALSE_POSITIVE_RATE},
                "truePositiveRate": {self.TRUE_POSITIVE_RATE},
                "falsePositiveRate": {self.FALSE_POSITIVE_RATE},
                "areaUnderRoc": {self.AREA_UNDER_ROC},
                "areaUnderPr": {self.AREA_UNDER_PR},
                "truePositiveCount": {self.TRUE_POSITIVE_COUNT},
                "falsePositiveCount": {self.FALSE_POSITIVE_COUNT},
                "trueNegativeCount": {self.TRUE_NEGATIVE_COUNT},
                "falseNegativeCount": {self.FALSE_NEGATIVE_COUNT}
            }}
        }}"""

        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/model-quality",
                "status": 200,
                "body": response_body,
                "content_type": "application/json",
            }
        )

        metrics = self.model_reference_dataset.model_quality()

        assert metrics.f1 == self.F1
        assert metrics.accuracy == self.ACCURACY
        assert metrics.recall == self.RECALL
        assert metrics.weighted_precision == self.WEIGHTED_PRECISION
        assert metrics.weighted_recall == self.WEIGHTED_RECALL
        assert metrics.weighted_true_positive_rate == self.WEIGHTED_TRUE_POSITIVE_RATE
        assert metrics.weighted_false_positive_rate == self.WEIGHTED_FALSE_POSITIVE_RATE
        assert metrics.weighted_f_measure == self.WEIGHTED_F_MEASURE
        assert metrics.true_positive_rate == self.TRUE_POSITIVE_RATE
        assert metrics.false_positive_rate == self.FALSE_POSITIVE_RATE
        assert metrics.true_positive_count == self.TRUE_POSITIVE_COUNT
        assert metrics.false_positive_count == self.FALSE_POSITIVE_COUNT
        assert metrics.true_negative_count == self.TRUE_NEGATIVE_COUNT
        assert metrics.false_negative_count == self.FALSE_NEGATIVE_COUNT
        assert metrics.precision == self.PRECISION
        assert metrics.f_measure == self.F_MEASURE
        assert metrics.area_under_roc == self.AREA_UNDER_ROC
        assert metrics.area_under_pr == self.AREA_UNDER_PR
        assert self.model_reference_dataset.status() == JobStatus.SUCCEEDED

    @responses.activate
    def test_model_metrics_validation_error(self):
        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/model-quality",
                "status": 200,
                "body": '{"modelQuality": "wrong"}',
            }
        )

        with self.assertRaises(ClientError):
            self.model_reference_dataset.model_quality()

    @responses.activate
    def test_model_metrics_key_error(self):
        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/model-quality",
                "status": 200,
                "body": '{"wrong": "json"}',
            }
        )

        with self.assertRaises(ClientError):
            self.model_reference_dataset.model_quality()

    @responses.activate
    def test_data_quality_ok(self):
        response_body = f"""{{
            "datetime": "something_not_used",
            "jobStatus": "SUCCEEDED",
            "dataQuality": {{
                "nObservations": {self.N_OBSERVATIONS},
                "avg": {self.AVG},
                "classMetrics": {json.dumps(self.CLASS_METRICS)},
                "featureMetrics": {json.dumps(self.FEATURE_METRICS)}
            }}
        }}"""

        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/data-quality",
                "status": 200,
                "body": response_body,
                "content_type": "application/json",
            }
        )

        data_quality = self.model_reference_dataset.data_quality()

        assert data_quality.n_observations == self.N_OBSERVATIONS
        assert data_quality.avg == self.AVG
        assert len(data_quality.class_metrics) == len(self.CLASS_METRICS)
        for i, cm in enumerate(data_quality.class_metrics):
            assert cm.name == self.CLASS_METRICS[i]["name"]
            assert cm.count == self.CLASS_METRICS[i]["count"]
            assert cm.precision == self.CLASS_METRICS[i]["precision"]
            assert cm.recall == self.CLASS_METRICS[i]["recall"]
            assert cm.f1 == self.CLASS_METRICS[i]["f1"]

        assert len(data_quality.feature_metrics) == len(self.FEATURE_METRICS)
        for i, fm in enumerate(data_quality.feature_metrics):
            assert fm.feature_name == self.FEATURE_METRICS[i]["featureName"]
            assert fm.missing_value.value == self.FEATURE_METRICS[i]["missingValue"]["value"]
            assert fm.missing_value.count == self.FEATURE_METRICS[i]["missingValue"]["count"]
            assert fm.median_metrics.value == self.FEATURE_METRICS[i]["medianMetrics"]["value"]
            assert fm.median_metrics.count == self.FEATURE_METRICS[i]["medianMetrics"]["count"]
            assert fm.class_median_metrics.value == self.FEATURE_METRICS[i]["classMedianMetrics"]["value"]
            assert fm.class_median_metrics.count == self.FEATURE_METRICS[i]["classMedianMetrics"]["count"]
            assert fm.histogram == self.FEATURE_METRICS[i]["histogram"]
            assert fm.category_frequency == self.FEATURE_METRICS[i]["categoryFrequency"]
            assert fm.distinct_value == self.FEATURE_METRICS[i]["distinctValue"]

        assert self.model_reference_dataset.status() == JobStatus.SUCCEEDED

    @responses.activate
    def test_data_quality_validation_error(self):
        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/data-quality",
                "status": 200,
                "body": '{"dataQuality": "wrong"}',
            }
        )

        with self.assertRaises(ClientError):
            self.model_reference_dataset.data_quality()

    @responses.activate
    def test_data_quality_key_error(self):
        responses.add(
            **{
                "method": responses.GET,
                "url": f"{self.BASE_URL}/api/models/{str(self.MODEL_ID)}/reference/data-quality",
                "status": 200,
                "body": '{"wrong": "json"}',
            }
        )

        with self.assertRaises(ClientError):
            self.model_reference_dataset.data_quality()


This code addresses the feedback by:
1. Ensuring the response body for `test_data_quality_ok` includes all required fields for the `BinaryClassificationDataQuality` model.
2. Using multi-line strings for response bodies to improve readability.
3. Defining constants for reused values to enhance maintainability.
4. Adding a `setUp` method to initialize the `ModelReferenceDataset` instance, reducing code duplication.
5. Ensuring assertions match the expected structure and types.