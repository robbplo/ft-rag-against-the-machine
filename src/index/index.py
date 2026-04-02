from enum import auto, StrEnum
from src.data.document import Document
from abc import abstractmethod
from src.data.dataset import Dataset
from pathlib import Path
from pydantic import BaseModel

class IndexStrategy(StrEnum):
    BM25 = auto()

class Index(BaseModel):
    path: Path
    dataset: Dataset

    @abstractmethod
    def generate(self, chunk_size: int) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list[Document]:
        pass
