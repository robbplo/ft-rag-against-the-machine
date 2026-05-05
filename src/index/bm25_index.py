from pathlib import Path
from src.models import MinimalSource
from bm25s.tokenization import tokenize
from bm25s import BM25
from src.index.index import IndexStrategy
from pydantic import PrivateAttr
import Stemmer  # ty:ignore[unresolved-import]
import json

class BM25IndexStrategy(IndexStrategy):
    _retriever: BM25 | None = PrivateAttr(default=None)
    _sources: list[MinimalSource] | None = PrivateAttr(default=None)
    _stemmer: Stemmer.Stemmer = PrivateAttr(default_factory=lambda: Stemmer.Stemmer("english"))

    def generate(self, chunk_size: int, sources: list[MinimalSource]) -> None:
        corpus = [src.content for src in sources]
        corpus_tokens = tokenize(corpus, stemmer=self._stemmer)
        retriever = BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.path)
        sources_path = self._sources_path()
        sources_path.write_text(json.dumps([src.model_dump(mode="json") for src in sources]))

    def load(self) -> None:
        self._retriever = BM25.load(str(self.path))
        sources_path = self._sources_path()
        self._sources = [MinimalSource.model_validate(d) for d in json.loads(sources_path.read_text())]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        if self._retriever is None or self._sources is None:
            raise ValueError("Index is not loaded")
        query_tokens = tokenize([query], stemmer=self._stemmer)
        results, _ = self._retriever.retrieve(query_tokens, corpus=self._sources, k=k)
        return list(results[0])
    
    def _sources_path(self) -> Path:
        return self.path / "sources.json"
