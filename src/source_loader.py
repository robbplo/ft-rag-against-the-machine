from src.models import MinimalSource
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class SourceLoader:
    def __init__(self, path: Path = Path("data/raw/vllm-0.10.1")):
        self.path = path

    def getSources(self, chunk_size: int) -> list[MinimalSource]:
        return self.getCode(chunk_size) + self.getDocs(chunk_size)

    def getCode(self, chunk_size: int) -> list[MinimalSource]:
        glob = Path.glob(self.path, "**/[!.]*.py")
        splitter = RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size*0.1
                )
        sources: list[MinimalSource] = []
        for path in glob:
            with open(path) as file:
                content = file.read()
            for chunk in splitter.create_documents([content]):
                start = content.index(chunk.page_content)
                sources.append(MinimalSource(
                    file_path=path,
                    content=chunk.page_content,
                    first_character_index=start,
                    last_character_index=start + len(chunk.page_content),
                ))
        return sources

    def getDocs(self, chunk_size: int) -> list[MinimalSource]:
        glob = Path.glob(self.path, "**/[!.]*.md")
        splitter = RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size*0.1
                )
        sources: list[MinimalSource] = []
        for path in glob:
            with open(path) as file:
                content = file.read()
            for chunk in splitter.create_documents([content]):
                start = content.index(chunk.page_content)
                sources.append(MinimalSource(
                    file_path=path,
                    content=chunk.page_content,
                    first_character_index=start,
                    last_character_index=start + len(chunk.page_content),
                ))
        return sources
