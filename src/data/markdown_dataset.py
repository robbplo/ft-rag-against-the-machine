from src.data.document import Document
from src.data.dataset import Dataset

class MarkdownDataset(Dataset):
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
