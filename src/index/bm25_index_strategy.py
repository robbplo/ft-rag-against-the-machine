"""BM25-backed retrieval index."""

import json
from pathlib import Path
import Stemmer  # ty:ignore[unresolved-import]
from bm25s import BM25
from bm25s.tokenization import tokenize
from pydantic import PrivateAttr
from src.index.index_strategy import IndexStrategy
from src.models import MinimalSource


class BM25IndexStrategy(IndexStrategy):
    """Persist and search a BM25 index with its source metadata."""

    _retriever: BM25 | None = PrivateAttr(default=None)
    _sources: list[MinimalSource] | None = PrivateAttr(default=None)
    _stemmer: Stemmer.Stemmer = PrivateAttr(
        default_factory=lambda: Stemmer.Stemmer("english")
    )

    def generate(
        self,
        chunk_size: int,
        sources: list[MinimalSource],
    ) -> None:
        """Build and persist the BM25 index and source metadata."""
        corpus = [source.content for source in sources]
        corpus_tokens = tokenize(corpus, stemmer=self._stemmer)
        retriever = BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.path)
        self._sources_path().write_text(
            json.dumps(
                [source.model_dump(mode="json") for source in sources]
            )
        )
        self._retriever = retriever
        self._sources = sources

    def load(self) -> None:
        """Load the persisted index and source metadata."""
        self._retriever = BM25.load(str(self.path))
        self._sources = [
            MinimalSource.model_validate(source)
            for source in json.loads(self._sources_path().read_text())
        ]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k indexed sources for a query."""
        if self._retriever is None or self._sources is None:
            raise ValueError("Index is not loaded")
        query_tokens = tokenize([query], stemmer=self._stemmer)
        results, _ = self._retriever.retrieve(
            query_tokens,
            corpus=self._sources,
            k=min(k, len(self._sources)),
            show_progress=False,
        )
        return list(results[0])

    def _sources_path(self) -> Path:
        return self.path / "sources.json"
