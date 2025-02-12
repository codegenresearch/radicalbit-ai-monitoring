from .aws_credentials import AwsCredentials
from .column_definition import ColumnDefinition
from .data_type import DataType
from .dataset_data_quality import (
    ClassificationDataQuality,
    CategoricalFeatureMetrics,
    CategoryFrequency,
    ClassMedianMetrics,
    ClassMetrics,
    DataQuality,
    FeatureMetrics,
    MedianMetrics,
    MissingValue,
    NumericalFeatureMetrics,
    RegressionDataQuality,
)
from .dataset_drift import (
    Drift,
    DriftAlgorithm,
    FeatureDrift,
    FeatureDriftCalculation,
)
from .dataset_model_quality import (
    BinaryClassificationModelQuality,
    CurrentBinaryClassificationModelQuality,
    CurrentMultiClassificationModelQuality,
    CurrentRegressionModelQuality,
    ModelQuality,
    MultiClassificationModelQuality,
    RegressionModelQuality,
)
from .dataset_stats import DatasetStats
from .field_type import FieldType
from .file_upload_result import CurrentFileUpload, FileReference, ReferenceFileUpload
from .job_status import JobStatus
from .model_definition import (
    CreateModel,
    Granularity,
    ModelDefinition,
    OutputType,
)
from .model_type import ModelType
from .supported_types import SupportedTypes

__all__ = [
    'OutputType',
    'Granularity',
    'CreateModel',
    'ModelDefinition',
    'ColumnDefinition',
    'JobStatus',
    'DataType',
    'ModelType',
    'DatasetStats',
    'ModelQuality',
    'BinaryClassificationModelQuality',
    'CurrentBinaryClassificationModelQuality',
    'CurrentMultiClassificationModelQuality',
    'MultiClassificationModelQuality',
    'RegressionModelQuality',
    'CurrentRegressionModelQuality',
    'DataQuality',
    'ClassificationDataQuality',
    'RegressionDataQuality',
    'ClassMetrics',
    'MedianMetrics',
    'MissingValue',
    'ClassMedianMetrics',
    'FeatureMetrics',
    'NumericalFeatureMetrics',
    'CategoryFrequency',
    'CategoricalFeatureMetrics',
    'DriftAlgorithm',
    'FeatureDriftCalculation',
    'FeatureDrift',
    'Drift',
    'ReferenceFileUpload',
    'CurrentFileUpload',
    'FileReference',
    'AwsCredentials',
    'SupportedTypes',
    'FieldType',
]

# Example of dynamically adding model features
def add_model_features(model_definition: ModelDefinition, features: list[ColumnDefinition]):
    model_definition.features.extend(features)
    return model_definition

# Example usage for API testing
def test_model_definition():
    feature1 = ColumnDefinition(name="feature1", type=SupportedTypes.float, field_type=FieldTypes.numerical)
    feature2 = ColumnDefinition(name="feature2", type=SupportedTypes.string, field_type=FieldTypes.categorical)
    model_def = ModelDefinition(
        uuid="12345678-1234-5678-1234-567812345678",
        name="Test Model",
        description="A test model for API testing",
        model_type=ModelType.BINARY,
        data_type=DataType.TABULAR,
        granularity=Granularity.DAY,
        features=[],
        outputs=OutputType(prediction=ColumnDefinition(name="prediction", type=SupportedTypes.bool, field_type=FieldTypes.categorical)),
        target=ColumnDefinition(name="target", type=SupportedTypes.bool, field_type=FieldTypes.categorical),
        timestamp=ColumnDefinition(name="timestamp", type=SupportedTypes.datetime, field_type=FieldTypes.datetime),
        frameworks="scikit-learn",
        algorithm="logistic regression",
        created_at="2023-10-01T00:00:00Z",
        updated_at="2023-10-01T00:00:00Z"
    )
    updated_model_def = add_model_features(model_def, [feature1, feature2])
    assert len(updated_model_def.features) == 2
    print("Model features added successfully for API testing.")