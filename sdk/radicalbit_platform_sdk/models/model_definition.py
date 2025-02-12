from enum import Enum
from typing import List, Optional
import uuid as uuid_lib

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from radicalbit_platform_sdk.models.column_definition import ColumnDefinition
from radicalbit_platform_sdk.models.data_type import DataType
from radicalbit_platform_sdk.models.model_type import ModelType


class OutputType(BaseModel):
    prediction: ColumnDefinition
    prediction_proba: Optional[ColumnDefinition] = None
    output: List[ColumnDefinition]

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class Granularity(str, Enum):
    HOUR = 'HOUR'
    DAY = 'DAY'
    WEEK = 'WEEK'
    MONTH = 'MONTH'


class BaseModelDefinition(BaseModel):
    """Base class for model definitions.\n\n    Attributes:\n        name: The name of the model.\n        description: An optional description of the model.\n        model_type: The type of the model.\n        data_type: The data type used by the model.\n        granularity: The window used for aggregated metrics.\n        features: A list of column definitions representing the features.\n        outputs: An OutputType definition explaining the model's output.\n        target: The column definition representing the model's target.\n        timestamp: The column definition for storing prediction timestamps.\n        frameworks: An optional field for describing the frameworks used by the model.\n        algorithm: An optional field for explaining the algorithm used by the model.\n    """

    name: str
    description: Optional[str] = None
    model_type: ModelType
    data_type: DataType
    granularity: Granularity
    features: List[ColumnDefinition]
    outputs: OutputType
    target: ColumnDefinition
    timestamp: ColumnDefinition
    frameworks: Optional[str] = None
    algorithm: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )

    def get_numerical_features(self) -> List[ColumnDefinition]:
        """Retrieve numerical features from the model's features."""
        return [feature for feature in self.features if feature.is_numerical()]

    def get_float_features(self) -> List[ColumnDefinition]:
        """Retrieve float features from the model's features."""
        return [feature for feature in self.features if feature.is_float()]

    def get_int_features(self) -> List[ColumnDefinition]:
        """Retrieve integer features from the model's features."""
        return [feature for feature in self.features if feature.is_int()]

    def get_categorical_features(self) -> List[ColumnDefinition]:
        """Retrieve categorical features from the model's features."""
        return [feature for feature in self.features if feature.is_categorical()]

    def get_datetime_features(self) -> List[ColumnDefinition]:
        """Retrieve datetime features from the model's features."""
        return [feature for feature in self.features if feature.is_datetime()]


class CreateModel(BaseModelDefinition):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ModelDefinition(BaseModelDefinition):
    uuid: uuid_lib.UUID = Field(default_factory=lambda: uuid_lib.uuid4())
    created_at: str = Field(alias='createdAt')
    updated_at: str = Field(alias='updatedAt')

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)