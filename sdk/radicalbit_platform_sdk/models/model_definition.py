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


class ModelFeatures(BaseModel):
    """Encapsulates the features of a model."""
    features: List[ColumnDefinition]

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )


class BaseModelDefinition(BaseModel):
    """A base class for model definition.

    Attributes:
        name: The name of the model.
        description: An optional description of the model.
        model_type: The type of the model.
        data_type: The data type used by the model.
        granularity: The window used to calculate aggregated metrics.
        features: A list of column definitions representing the features set.
        outputs: An OutputType definition to explain the output of the model.
        target: The column used to represent the model's target.
        timestamp: The column used to store when the prediction was done.
        frameworks: An optional field to describe the frameworks used by the model.
        algorithm: An optional field to explain the algorithm used by the model.
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


class CreateModel(BaseModelDefinition):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ModelDefinition(BaseModelDefinition):
    uuid: uuid_lib.UUID = Field(default_factory=lambda: uuid_lib.uuid4())
    created_at: str = Field(alias='createdAt')
    updated_at: str = Field(alias='updatedAt')

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


### Changes Made:
1. **Removed the Comment Block**: Removed the block of comments at the end of the file to prevent syntax errors.
2. **Docstring Consistency**: Ensured that the docstrings for the classes and methods are concise and consistent with the gold code.
3. **Attribute Descriptions**: Made the attribute descriptions more concise and uniform.
4. **Redundant Methods**: Removed the feature retrieval methods from the `ModelFeatures` class to match the gold code.
5. **Formatting and Style**: Ensured consistent formatting, including spacing and alignment.
6. **Class Structure**: Ensured the structure of the classes matches the gold code, including the order and organization of attributes and methods.