from src.models import MinimalSource
from enum import auto, StrEnum
from abc import abstractmethod
from src.source_loader import SourceLoader
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
