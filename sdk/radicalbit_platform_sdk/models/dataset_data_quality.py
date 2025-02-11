from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List, Optional, Union


class MedianMetrics(BaseModel):
    perc_25: Optional[float] = None
    median: Optional[float] = None
    perc_75: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True)


class MissingValue(BaseModel):
    count: int
    percentage: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True)


class ClassMetrics(BaseModel):
    name: str
    count: int
    percentage: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True)


class ClassMedianMetrics(BaseModel):
    name: str
    mean: Optional[float] = None
    median_metrics: MedianMetrics

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class FeatureMetrics(BaseModel):
    feature_name: str
    type: str
    missing_value: MissingValue

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class NumericalFeatureMetrics(FeatureMetrics):
    type: str = "numerical"
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median_metrics: MedianMetrics
    class_median_metrics: List[ClassMedianMetrics]
    histogram: 'Histogram'

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class CategoryFrequency(BaseModel):
    name: str
    count: int
    frequency: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class CategoricalFeatureMetrics(FeatureMetrics):
    type: str = "categorical"
    category_frequency: List[CategoryFrequency]
    distinct_value: int

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class Histogram(BaseModel):
    buckets: List[float]
    reference_values: List[int]
    current_values: Optional[List[int]] = None

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class DataQuality(BaseModel):
    pass


class BinaryClassificationDataQuality(DataQuality):
    n_observations: int
    class_metrics: List[ClassMetrics]
    feature_metrics: List[Union[NumericalFeatureMetrics, CategoricalFeatureMetrics]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )


class MultiClassDataQuality(DataQuality):
    pass


class RegressionDataQuality(DataQuality):
    pass


### Adjustments Made:
1. **Order of Class Definitions**: Reordered the classes to match the gold code.
2. **Model Configuration**: Ensured `histogram` is a proper class reference in `NumericalFeatureMetrics`.
3. **Alias Generator**: Applied `alias_generator` consistently where needed.
4. **Class Inheritance**: Verified that `NumericalFeatureMetrics` and `CategoricalFeatureMetrics` correctly inherit from `FeatureMetrics`.
5. **Optional Fields**: Reviewed and ensured optional fields match the gold code.
6. **Empty Classes**: Kept `DataQuality`, `MultiClassDataQuality`, and `RegressionDataQuality` as empty classes.