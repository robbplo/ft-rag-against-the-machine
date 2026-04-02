from src.data.document import Document
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class Dataset(BaseModel):
    path: Path

    def getDocuments(self, chunk_size: int) -> list[Document]:
        return self.getCode(chunk_size) + self.getDocs(chunk_size)

    def getCode(self, chunk_size: int) -> list[Document]:
        glob = Path.glob(self.path, "**/[!.]*.py")
        splitter = RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size*0.1
                )
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

    def getDocs(self, chunk_size: int) -> list[Document]:
        glob = Path.glob(self.path, "**/[!.]*.md")
        splitter = RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size*0.1
                )
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
