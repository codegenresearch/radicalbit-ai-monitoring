from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Optional


class DatasetStats(BaseModel):
    pass


class BinaryClassificationStats(DatasetStats):
    f1: float
    accuracy: float
    precision: float
    recall: float
    area_under_roc: Optional[float] = None
    area_under_pr: Optional[float] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel, protected_namespaces=()
    )