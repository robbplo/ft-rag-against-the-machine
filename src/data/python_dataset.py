from src.data.dataset import Dataset
from src.data.document import Document
from pydantic import BaseModel, Field
from pathlib import Path

class PythonDataset(Dataset):
    def getDocuments(self) -> list[Document]:
        glob = Path.glob(self.path, "**/[!.]*.py")
        documents: list[Document] = []
        for path in glob:
            content = ""
            with open(path) as file:
                content = file.read()
            document = Document(path=path, content=content, start_index=0, end_index=len(content))
            documents.append(document)
        return documents
