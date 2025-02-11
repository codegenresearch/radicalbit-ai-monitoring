from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List, Optional


class ClassMetrics(BaseModel):
    name: str
    count: int
    percentage: float = 0.0

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class MedianMetrics(BaseModel):
    perc_25: float = 0.0
    median: float = 0.0
    perc_75: float = 0.0

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class MissingValue(BaseModel):
    count: int = 0
    percentage: float = 0.0

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class ClassMedianMetrics(BaseModel):
    name: str
    mean: float = 0.0
    median_metrics: MedianMetrics

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class FeatureMetrics(BaseModel):
    feature_name: str
    type: str
    missing_value: MissingValue

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class NumericalFeatureMetrics(FeatureMetrics):
    type: str = "numerical"
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median_metrics: MedianMetrics
    class_median_metrics: List[ClassMedianMetrics]

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class CategoryFrequency(BaseModel):
    name: str
    count: int = 0
    frequency: float = 0.0

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class CategoricalFeatureMetrics(FeatureMetrics):
    type: str = "categorical"
    category_frequency: List[CategoryFrequency]
    distinct_value: int = 0

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class DataQuality(BaseModel):
    pass


class BinaryClassificationDataQuality(DataQuality):
    n_observations: int = 0
    class_metrics: List[ClassMetrics] = []
    feature_metrics: List[FeatureMetrics] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        alias_generator=to_camel,
        protected_namespaces=(),
    )


class MultiClassDataQuality(DataQuality):
    pass


class RegressionDataQuality(DataQuality):
    pass