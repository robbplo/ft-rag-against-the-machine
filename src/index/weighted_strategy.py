from src.index.index_strategy import IndexStrategy
from pydantic import BaseModel, Field


class WeightedStrategy(BaseModel):
    index: IndexStrategy
    weight: float = Field(gt=0, le=1)
