from src.index.index_strategy import IndexStrategy
from pydantic import BaseModel, Field


class WeightedStrategy(BaseModel):
    """Pair an index strategy with its reciprocal-rank fusion weight."""

    index: IndexStrategy
    weight: float = Field(gt=0, le=1)
