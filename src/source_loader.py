"""Load supported source and documentation files into retrieval chunks."""

from pathlib import Path
from typing import ClassVar
import warnings

from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import MinimalSource


class SourceLoader:
    """Discover and chunk relevant text files from a source tree."""

    CODE_LANGUAGES: ClassVar[dict[str, Language]] = {
        ".c": Language.C,
        ".cc": Language.CPP,
        ".cpp": Language.CPP,
        ".cu": Language.CPP,
        ".cuh": Language.CPP,
        ".cxx": Language.CPP,
        ".h": Language.C,
        ".hh": Language.CPP,
        ".hpp": Language.CPP,
        ".hxx": Language.CPP,
        ".inl": Language.CPP,
        ".js": Language.JS,
        ".py": Language.PYTHON,
    }
    DOCUMENT_LANGUAGES: ClassVar[dict[str, Language]] = {
        ".html": Language.HTML,
        ".md": Language.MARKDOWN,
        ".rst": Language.RST,
    }
    TEXT_SUFFIXES: ClassVar[frozenset[str]] = frozenset({
        ".cmake",
        ".css",
        ".env",
        ".in",
        ".jinja",
        ".json",
        ".jsonl",
        ".patch",
        ".sh",
        ".toml",
        ".tpl",
        ".txt",
        ".yaml",
        ".yml",
    })
    TEXT_FILENAMES: ClassVar[frozenset[str]] = frozenset({
        ".clang-format",
        ".dockerignore",
        ".gitignore",
        ".helmignore",
        ".shellcheckrc",
        ".yapfignore",
        "CODEOWNERS",
        "DCO",
        "LICENSE",
        "Makefile",
        "README",
    })

    def __init__(self, path: Path = Path("data/raw/vllm-0.10.1")):
        """Configure the root directory containing the source corpus."""
        self.path = path

    def getSources(self, chunk_size: int) -> list[MinimalSource]:
        """Return code chunks followed by documentation and text chunks."""
        return self.getCode(chunk_size) + self.getDocs(chunk_size)

    def getCode(self, chunk_size: int) -> list[MinimalSource]:
        """Load programming-language files with language-aware splitting."""
        splitters = {
            language: self._splitter(chunk_size, language)
            for language in set(self.CODE_LANGUAGES.values())
        }
        sources: list[MinimalSource] = []
        for path in self._source_paths():
            language = self.CODE_LANGUAGES.get(path.suffix.lower())
            if language is None:
                continue
            sources.extend(self._chunks(path, splitters[language]))
        return sources

    def getDocs(self, chunk_size: int) -> list[MinimalSource]:
        """Load documentation, configuration, and other plain-text files."""
        language_splitters = {
            language: self._splitter(chunk_size, language)
            for language in set(self.DOCUMENT_LANGUAGES.values())
        }
        text_splitter = self._splitter(chunk_size)
        sources: list[MinimalSource] = []
        for path in self._source_paths():
            suffix = path.suffix.lower()
            language = self.DOCUMENT_LANGUAGES.get(suffix)
            if language is not None:
                splitter = language_splitters[language]
            elif self._is_plain_text(path):
                splitter = text_splitter
            else:
                continue
            sources.extend(self._chunks(path, splitter))
        return sources

    def _source_paths(self) -> list[Path]:
        """Return deterministic files while excluding nested Git metadata."""
        return sorted(
            (
                path
                for path in self.path.rglob("*")
                if path.is_file()
                and ".git" not in path.relative_to(self.path).parts
            ),
            key=lambda path: path.as_posix(),
        )

    def _is_plain_text(self, path: Path) -> bool:
        """Return whether a path is an allowlisted text or build file."""
        return (
            path.suffix.lower() in self.TEXT_SUFFIXES
            or path.name in self.TEXT_FILENAMES
            or path.name == "Dockerfile"
            or path.name.startswith("Dockerfile.")
        )

    def _splitter(
        self,
        chunk_size: int,
        language: Language | None = None,
    ) -> RecursiveCharacterTextSplitter:
        """Create a splitter that records exact source character offsets."""
        if language is None:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 10,
                add_start_index=True,
            )
        return RecursiveCharacterTextSplitter.from_language(
            language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10,
            add_start_index=True,
        )

    def _chunks(
        self,
        path: Path,
        splitter: RecursiveCharacterTextSplitter,
    ) -> list[MinimalSource]:
        """Read one text file and convert it to offset-preserving chunks."""
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            warnings.warn(
                f"Skipping unreadable source {path}: {error}",
                stacklevel=2,
            )
            return []

        sources: list[MinimalSource] = []
        for chunk in splitter.create_documents([content]):
            start = chunk.metadata.get("start_index")
            if not isinstance(start, int):
                raise ValueError(f"Chunk is missing a start index for {path}")
            sources.append(
                MinimalSource(
                    file_path=path,
                    content=chunk.page_content,
                    first_character_index=start,
                    last_character_index=start + len(chunk.page_content),
                )
            )
        return sources
