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
    """Base model definition class.

    Attributes:
        name: Name of the model.
        description: Optional description of the model.
        model_type: Type of the model.
        data_type: Data type used by the model.
        granularity: Granularity for aggregated metrics.
        features: List of column definitions representing the features.
        outputs: Definition of the model's output.
        target: Column definition for the model's target.
        timestamp: Column definition for storing prediction timestamps.
        frameworks: Optional description of frameworks used.
        algorithm: Optional description of the algorithm used.
    """

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
        """Retrieve all numerical features from the model."""
        return [feature for feature in self.features if feature.is_numerical()]

    def get_float_features(self) -> List[ColumnDefinition]:
        """Retrieve all float features from the model."""
        return [feature for feature in self.features if feature.is_float()]

    def get_int_features(self) -> List[ColumnDefinition]:
        """Retrieve all integer features from the model."""
        return [feature for feature in self.features if feature.is_int()]

    def get_categorical_features(self) -> List[ColumnDefinition]:
        """Retrieve all categorical features from the model."""
        return [feature for feature in self.features if feature.is_categorical()]

    def get_datetime_features(self) -> List[ColumnDefinition]:
        """Retrieve all datetime features from the model."""
        return [feature for feature in self.features if feature.is_datetime()]


class CreateModel(BaseModelDefinition):
    """Model definition for creating a new model."""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ModelDefinition(BaseModelDefinition):
    """Model definition including UUID, creation, and update timestamps."""
    uuid: uuid_lib.UUID = Field(default_factory=uuid_lib.uuid4)
    created_at: str = Field(alias='createdAt')
    updated_at: str = Field(alias='updatedAt')

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)