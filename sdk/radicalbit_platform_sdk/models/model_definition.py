from enum import Enum
from typing import List, Optional
import uuid as uuid_lib

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
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
    """A base class for model definition.\n\n    Attributes:\n        name: The name of the model.\n        description: An optional description to explain something about the model.\n        model_type: The type of the model\n        data_type: It explains the data type used by the model\n        granularity: The window used to calculate aggregated metrics\n        features: A list column representing the features set\n        outputs: An OutputType definition to explain the output of the model\n        target: The column used to represent model's target\n        timestamp: The column used to store when prediction was done\n        frameworks: An optional field to describe the frameworks used by the model\n        algorithm: An optional field to explain the algorithm used by the model\n\n    """

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
        return TypeAdapter(List[ColumnDefinition]).validate_python(
            [feature for feature in self.features if feature.is_numerical()]
        )

    def get_float_features(self) -> List[ColumnDefinition]:
        return TypeAdapter(List[ColumnDefinition]).validate_python(
            [feature for feature in self.features if feature.is_float()]
        )

    def get_int_features(self) -> List[ColumnDefinition]:
        return TypeAdapter(List[ColumnDefinition]).validate_python(
            [feature for feature in self.features if feature.is_int()]
        )

    def get_categorical_features(self) -> List[ColumnDefinition]:
        return TypeAdapter(List[ColumnDefinition]).validate_python(
            [feature for feature in self.features if feature.is_categorical()]
        )


class CreateModel(BaseModelDefinition):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ModelDefinition(BaseModelDefinition):
    uuid: uuid_lib.UUID = Field(default_factory=lambda: uuid_lib.uuid4())
    created_at: str = Field(alias='createdAt')
    updated_at: str = Field(alias='updatedAt')

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


def test_feature_update():
    model_features = [
        ColumnDefinition(name="feature1", type="int", field_type="numerical"),
        ColumnDefinition(name="feature2", type="string", field_type="categorical"),
    ]
    model_definition = ModelDefinition(
        name="test_model",
        model_type=ModelType.BINARY,
        data_type=DataType.TABULAR,
        granularity=Granularity.DAY,
        features=model_features,
        outputs=OutputType(prediction=ColumnDefinition(name="prediction", type="float", field_type="numerical")),
        target=ColumnDefinition(name="target", type="bool", field_type="categorical"),
        timestamp=ColumnDefinition(name="timestamp", type="datetime", field_type="datetime"),
    )
    assert model_definition.features == model_features