from src.data.document import Document
from pydantic import BaseModel, Field
from pathlib import Path

class Dataset(BaseModel):
    path: Path

    def getDocuments(self) -> list[Document]:
        return []
