"""Load source files and split them into location-aware chunks."""

from collections.abc import Iterable
from pathlib import Path

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from tqdm import tqdm

from src.models import MinimalSource


class SourceLoader:
    """Discover and chunk supported code and documentation files."""

    def __init__(self, path: Path = Path("data/raw/vllm-0.10.1")) -> None:
        """Store the corpus root used for source discovery."""
        self.path = path

    def getSources(self, chunk_size: int) -> list[MinimalSource]:
        """Return all supported code and documentation chunks."""
        return self.getCode(chunk_size) + self.getDocs(chunk_size)

    def getCode(self, chunk_size: int) -> list[MinimalSource]:
        """Split Python files using language-aware separators."""
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            add_start_index=True,
        )
        return self._split_files(
            Path.glob(self.path, "**/[!.]*.py"),
            splitter,
            "Chunking Python",
        )

    def getDocs(self, chunk_size: int) -> list[MinimalSource]:
        """Split Markdown and plain-text files with distinct strategies."""
        overlap = int(chunk_size * 0.1)
        markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        markdown_sources = self._split_files(
            Path.glob(self.path, "**/[!.]*.md"),
            markdown_splitter,
            "Chunking Markdown",
        )
        text_sources = self._split_files(
            Path.glob(self.path, "**/[!.]*.txt"),
            text_splitter,
            "Chunking text",
        )
        return markdown_sources + text_sources

    def _split_files(
        self,
        paths: Iterable[Path],
        splitter: TextSplitter,
        description: str,
    ) -> list[MinimalSource]:
        """Split files while preserving each chunk's exact source offset."""
        source_paths = list(paths)
        sources: list[MinimalSource] = []
        for path in tqdm(source_paths, desc=description, unit="file"):
            content = path.read_text()
            for chunk in splitter.create_documents([content]):
                start = chunk.metadata.get("start_index")
                if not isinstance(start, int) or start < 0:
                    raise ValueError(
                        f"Could not determine chunk offset in {path}"
                    )
                end = start + len(chunk.page_content)
                if content[start:end] != chunk.page_content:
                    raise ValueError(f"Invalid chunk offset in {path}")
                sources.append(
                    MinimalSource(
                        file_path=str(path),
                        content=chunk.page_content,
                        first_character_index=start,
                        last_character_index=end,
                    )
                )
        return sources
