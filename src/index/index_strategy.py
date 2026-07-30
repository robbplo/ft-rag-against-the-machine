from src.models import MinimalSource
from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel


class IndexStrategy(BaseModel):
    path: Path

    @abstractmethod
    def generate(self, chunk_size: int, sources: list[MinimalSource]) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list[MinimalSource]:
        pass
