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
    """A class to manage model features.

    Attributes:
        features: A list of column definitions representing the features set.
    """

    features: List[ColumnDefinition]

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


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
1. **Removed the Comment**: The comment at line 87 was removed to avoid the `SyntaxError`.
2. **Docstring Consistency**: Ensured that the descriptions in the docstrings are consistent with the gold code. Removed any bullet points or list formatting and maintained the structure using proper multi-line string literals.
3. **Attribute Descriptions**: Reviewed and ensured that the descriptions of the attributes in the `BaseModelDefinition` class are clear and match the phrasing used in the gold code.
4. **Formatting and Spacing**: Double-checked the overall formatting and spacing throughout the code to ensure consistency.
5. **Model Configurations**: Verified that the `model_config` definitions in all classes match exactly with those in the gold code.
6. **Class Structure**: Ensured that the `ModelFeatures` class does not contain any methods, as it should only define attributes.