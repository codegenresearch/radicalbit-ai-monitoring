from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List, Optional, Union
from app.models.exceptions import MetricsInternalError


class ClassMetrics(BaseModel):
    name: str
    count: int
    percentage: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class MedianMetrics(BaseModel):
    perc_25: Optional[float] = None
    median: Optional[float] = None
    perc_75: Optional[float] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class MissingValue(BaseModel):
    count: int
    percentage: Optional[float] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class ClassMedianMetrics(BaseModel):
    name: str
    mean: Optional[float] = None
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
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median_metrics: MedianMetrics
    class_median_metrics: List[ClassMedianMetrics]

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class CategoryFrequency(BaseModel):
    name: str
    count: int
    frequency: Optional[float] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class CategoricalFeatureMetrics(FeatureMetrics):
    type: str = "categorical"
    category_frequency: List[CategoryFrequency]
    distinct_value: Optional[int] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class DataQuality(BaseModel):
    pass


class BinaryClassificationDataQuality(DataQuality):
    n_observations: Optional[int] = None
    class_metrics: Optional[List[ClassMetrics]] = None
    feature_metrics: Optional[List[FeatureMetrics]] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        alias_generator=to_camel,
        protected_namespaces=(),
    )

    @classmethod
    def from_dict(cls, data: dict) -> "BinaryClassificationDataQuality":
        try:
            return cls(**data)
        except Exception as e:
            raise MetricsInternalError(f"Error parsing BinaryClassificationDataQuality: {e}")


class MultiClassDataQuality(DataQuality):
    n_observations: Optional[int] = None
    class_metrics: Optional[List[ClassMetrics]] = None
    feature_metrics: Optional[List[FeatureMetrics]] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        alias_generator=to_camel,
        protected_namespaces=(),
    )

    @classmethod
    def from_dict(cls, data: dict) -> "MultiClassDataQuality":
        try:
            return cls(**data)
        except Exception as e:
            raise MetricsInternalError(f"Error parsing MultiClassDataQuality: {e}")


class RegressionDataQuality(DataQuality):
    n_observations: Optional[int] = None
    feature_metrics: Optional[List[FeatureMetrics]] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        alias_generator=to_camel,
        protected_namespaces=(),
    )

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionDataQuality":
        try:
            return cls(**data)
        except Exception as e:
            raise MetricsInternalError(f"Error parsing RegressionDataQuality: {e}")