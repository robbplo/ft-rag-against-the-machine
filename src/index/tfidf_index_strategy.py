"""TF-IDF-backed retrieval index."""
from typing import Any
from math import log10, sqrt
from collections import Counter
import json
from pathlib import Path
import Stemmer  # ty:ignore[unresolved-import]
from bm25s.tokenization import tokenize, Tokenized
from pydantic import PrivateAttr
from src.index.index_strategy import IndexStrategy
from src.models import MinimalSource


class TFIDFIndexStrategy(IndexStrategy):
    """Persist and search a TF-IDF index with its source metadata."""

    _sources: list[MinimalSource] | None = PrivateAttr(default=None)
    _vocab: dict[str, int] | None = PrivateAttr(default=None)
    _idf: dict[int, float] | None = PrivateAttr(default=None)
    _tfidf_dicts: list[dict[int, float]] | None = PrivateAttr(default=None)
    _document_norms: list[float] | None = PrivateAttr(default=None)
    _stemmer: Stemmer.Stemmer = PrivateAttr(
        default_factory=lambda: Stemmer.Stemmer("english")
    )

    def generate(
        self,
        chunk_size: int,
        sources: list[MinimalSource],
    ) -> None:
        """Build and persist the TF-IDF index and source metadata."""
        corpus = [source.content for source in sources]
        tokenized: Any = tokenize(
            corpus,
            stemmer=self._stemmer,
        )
        if not isinstance(tokenized, Tokenized):
            raise ValueError('Expected Tokenized instance')
        corpus_tokens: Tokenized = tokenized

        # Count word frequencies per document
        doc_term_frequencies: list[Counter[int]] = []
        corpus_frequency: Counter[int] = Counter()
        for doc_tokens in corpus_tokens.ids:
            doc_frequency = Counter(doc_tokens)
            doc_term_frequencies.append(doc_frequency)
            corpus_frequency.update(doc_frequency.keys())

        # Calculate IDF
        idf = {
            token: log10((len(sources) + 1.0) / (df + 1.0)) + 1.0
            for token, df in corpus_frequency.items()
        }

        # Calculate TD-IDF dicts
        tfidf_dicts: list[dict[int, float]] = []
        document_norms: list[float] = []
        for doc_frequency, doc_tokens in zip(
            doc_term_frequencies,
            corpus_tokens.ids
        ):
            doc_length = len(doc_tokens)

            vector: dict[int, float] = {}
            for token, token_count in doc_frequency.items():
                vector[token] = (token_count / doc_length) * idf[token]
            tfidf_dicts.append(vector)
            document_norms.append(
                sqrt(sum(w * w for w in vector.values()))
            )

        # Persist the complete state needed to score queries after loading.
        self.path.mkdir(parents=True, exist_ok=True)
        self._index_path().write_text(
            json.dumps(
                {
                    "vocab": corpus_tokens.vocab,
                    "idf": idf,
                    "tfidf_dicts": tfidf_dicts,
                    "document_norms": document_norms,
                }
            )
        )
        self._sources_path().write_text(
            json.dumps(
                [source.model_dump(mode="json") for source in sources]
            )
        )
        self._vocab = corpus_tokens.vocab
        self._idf = idf
        self._tfidf_dicts = tfidf_dicts
        self._document_norms = document_norms
        self._sources = sources

    def load(self) -> None:
        """Load the persisted index and source metadata."""
        state: dict[str, Any] = json.loads(self._index_path().read_text())
        vocab = {
            str(token): int(token_id)
            for token, token_id in state["vocab"].items()
        }
        idf = {
            int(token_id): float(weight)
            for token_id, weight in state["idf"].items()
        }
        tfidf_dicts = [
            {
                int(token_id): float(weight)
                for token_id, weight in vector.items()
            }
            for vector in state["tfidf_dicts"]
        ]
        document_norms = [
            float(norm) for norm in state["document_norms"]
        ]
        sources = [
            MinimalSource.model_validate(source)
            for source in json.loads(self._sources_path().read_text())
        ]

        if not (len(tfidf_dicts) == len(document_norms) == len(sources)):
            raise ValueError(
                "Persisted TF-IDF vectors, norms, and sources are misaligned"
            )

        self._vocab = vocab
        self._idf = idf
        self._tfidf_dicts = tfidf_dicts
        self._document_norms = document_norms
        self._sources = sources

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k indexed sources for a query."""
        if (
            self._sources is None
            or self._vocab is None
            or self._idf is None
            or self._tfidf_dicts is None
            or self._document_norms is None
        ):
            raise ValueError("Index is not loaded")
        if k <= 0:
            return []

        tokenized: Any = tokenize(
            [query],
            stemmer=self._stemmer,
            return_ids=False,
            show_progress=False
        )
        if isinstance(tokenized, Tokenized):
            raise ValueError("Expected string tokens")

        query_tokens: list[str] = tokenized[0]
        if not query_tokens:
            return []

        query_frequency = Counter(query_tokens)
        query_vector: dict[int, float] = {}
        for token, token_count in query_frequency.items():
            token_id = self._vocab.get(token)
            if token_id is None:
                continue
            query_vector[token_id] = (
                token_count / len(query_tokens)
            ) * self._idf[token_id]

        query_norm = sqrt(
            sum(weight * weight for weight in query_vector.values())
        )
        if query_norm == 0.0:
            return []

        scores: list[tuple[float, int]] = []
        for index, (document_vector, document_norm) in enumerate(
            zip(self._tfidf_dicts, self._document_norms)
        ):
            if document_norm == 0.0:
                score = 0.0
            else:
                dot_product = sum(
                    query_weight * document_vector.get(token_id, 0.0)
                    for token_id, query_weight in query_vector.items()
                )
                score = dot_product / (query_norm * document_norm)
            scores.append((score, index))

        scores.sort(key=lambda result: result[0], reverse=True)
        return [
            self._sources[index]
            for _, index in scores[:min(k, len(scores))]
        ]

    def _sources_path(self) -> Path:
        return self.path / "sources.json"

    def _index_path(self) -> Path:
        return self.path / "index.json"
