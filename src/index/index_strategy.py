from src.models import MinimalSource
from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel


class IndexStrategy(BaseModel):
    """Define the common interface for persisted retrieval indexes."""

    path: Path

    @abstractmethod
    def generate(self, chunk_size: int, sources: list[MinimalSource]) -> None:
        """Build and persist an index from source chunks."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load previously persisted index data."""
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to ``k`` source chunks relevant to a query."""
        pass
