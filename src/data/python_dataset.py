from src.data.dataset import Dataset
from src.data.document import Document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class PythonDataset(Dataset):
    def getDocuments(self) -> list[Document]:
        glob = Path.glob(self.path, "**/[!.]*.py")
        splitter = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=2000)
        documents: list[Document] = []
        for path in glob:
            with open(path) as file:
                content = file.read()
            for chunk in splitter.create_documents([content]):
                start = content.index(chunk.page_content)
                documents.append(Document(
                    path=path,
                    content=chunk.page_content,
                    start_index=start,
                    end_index=start + len(chunk.page_content),
                ))
        return documents
