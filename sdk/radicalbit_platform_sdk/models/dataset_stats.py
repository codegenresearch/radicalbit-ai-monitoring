from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Optional


class DatasetStats(BaseModel):
    n_variables: Optional[int] = None
    n_observations: Optional[int] = None
    missing_cells: Optional[int] = None
    missing_cells_perc: Optional[float] = None
    duplicate_rows: Optional[int] = None
    duplicate_rows_perc: Optional[float] = None
    numeric: Optional[int] = None
    categorical: Optional[int] = None
    datetime: Optional[int] = None

    model_config = ConfigDict(
        populate_by_name=True, alias_generator=to_camel
    )