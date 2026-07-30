"""Semantic vector retrieval index."""

import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, PrivateAttr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.index.index_strategy import IndexStrategy
from src.models import MinimalSource


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "mps"
EMBEDDING_QUERY_PROMPT_NAME = None


class SemanticIndexManifest(BaseModel):
    """Record which model generated the persisted embeddings."""

    model_name: str


class SemanticIndexStrategy(IndexStrategy):
    """Persist and search normalized sentence-transformer embeddings."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = Field(default=32, gt=0)

    _model: SentenceTransformer | None = PrivateAttr(default=None)
    _embeddings: NDArray[np.float32] | None = PrivateAttr(default=None)
    _sources: list[MinimalSource] | None = PrivateAttr(default=None)

    def generate(
        self,
        chunk_size: int,
        sources: list[MinimalSource],
    ) -> None:
        """Embed sources and persist vectors plus source metadata."""
        del chunk_size
        if not sources:
            raise ValueError("Cannot build a semantic index without sources")

        model = self._load_model()
        embedding_batches = []
        batch_starts = range(0, len(sources), self.batch_size)
        for start_index in tqdm(
            batch_starts,
            desc="Embedding sources",
            unit="batch",
        ):
            batch = sources[start_index:start_index + self.batch_size]
            embedding_batches.append(
                model.encode(
                    [source.content for source in batch],
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
        embeddings = np.concatenate(
            [
                np.asarray(batch, dtype=np.float32)
                for batch in embedding_batches
            ],
            axis=0,
        )
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self._embeddings_path(), embeddings, allow_pickle=False)
        self._sources_path().write_text(
            json.dumps(
                [source.model_dump(mode="json") for source in sources]
            )
        )
        self._manifest_path().write_text(
            SemanticIndexManifest(
                model_name=self.model_name,
            ).model_dump_json()
        )

        self._embeddings = embeddings
        self._sources = sources

    def load(self) -> None:
        """Load vectors, source metadata, and the encoder."""
        manifest = SemanticIndexManifest.model_validate_json(
            self._manifest_path().read_text()
        )
        if manifest.model_name != self.model_name:
            raise ValueError(
                "Semantic index model does not match the configured model; "
                "rebuild the index"
            )
        self._sources = [
            MinimalSource.model_validate(source)
            for source in json.loads(self._sources_path().read_text())
        ]
        self._embeddings = cast(
            NDArray[np.float32],
            np.load(
                self._embeddings_path(),
                allow_pickle=False,
                mmap_mode="r",
            ),
        )
        self._load_model()

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources by exact cosine similarity."""
        if self._model is None or self._embeddings is None:
            raise ValueError("Index is not loaded")
        if self._sources is None:
            raise ValueError("Index sources are not loaded")
        if k <= 0 or not query.strip() or not self._sources:
            return []

        query_embedding = np.asarray(
            self._model.encode(
                query,
                prompt_name=EMBEDDING_QUERY_PROMPT_NAME,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        scores = self._embeddings @ query_embedding
        ranked_indices = np.argsort(-scores, kind="stable")
        return [self._sources[int(index)] for index in ranked_indices[:k]]

    def _load_model(self) -> SentenceTransformer:
        """Load the configured encoder once."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=EMBEDDING_DEVICE,
            )
        return self._model

    def _embeddings_path(self) -> Path:
        """Return the path used to persist source embeddings."""
        return self.path / "embeddings.npy"

    def _sources_path(self) -> Path:
        """Return the path used to persist source metadata."""
        return self.path / "sources.json"

    def _manifest_path(self) -> Path:
        """Return the path used to persist index configuration metadata."""
        return self.path / "manifest.json"
