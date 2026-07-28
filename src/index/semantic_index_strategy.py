"""CPU-backed semantic vector retrieval index."""

import json
from pathlib import Path
from typing import cast

import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from numpy.typing import NDArray
from pydantic import BaseModel, Field, PrivateAttr
from sentence_transformers import SentenceTransformer

from src.index.index_strategy import IndexStrategy
from src.models import MinimalSource


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_INDEX_FORMAT_VERSION = 1
MODEL_FILE_PATTERNS = [
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


class SemanticIndexManifest(BaseModel):
    """Describe the persisted vector index format and encoder."""

    format_version: int = SEMANTIC_INDEX_FORMAT_VERSION
    model_name: str
    dimensions: int = Field(gt=0)
    source_count: int = Field(ge=0)
    normalized: bool


class SemanticIndexStrategy(IndexStrategy):
    """Persist and search dense MiniLM embeddings using exact cosine search."""

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
        """Embed sources on CPU and persist vectors plus source metadata."""
        del chunk_size
        if not sources:
            raise ValueError("Cannot build a semantic index without sources")

        model = self._load_model()
        raw_embeddings = model.encode(
            [source.content for source in sources],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(sources):
            raise ValueError("Encoder returned an invalid embedding matrix")

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
                dimensions=embeddings.shape[1],
                source_count=len(sources),
                normalized=True,
            ).model_dump_json()
        )

        self._embeddings = embeddings
        self._sources = sources

    def load(self) -> None:
        """Load and validate vectors, metadata, and the CPU encoder."""
        manifest_text = self._manifest_path().read_text()
        manifest = SemanticIndexManifest.model_validate_json(manifest_text)
        if manifest.format_version != SEMANTIC_INDEX_FORMAT_VERSION:
            raise ValueError(
                "Unsupported semantic index format version: "
                f"{manifest.format_version}"
            )
        if manifest.model_name != self.model_name:
            raise ValueError(
                "Semantic index model does not match the configured model; "
                "rebuild the index"
            )
        if not manifest.normalized:
            raise ValueError("Semantic index vectors must be normalized")

        sources = [
            MinimalSource.model_validate(source)
            for source in json.loads(self._sources_path().read_text())
        ]
        embeddings = cast(
            NDArray[np.float32],
            np.load(
                self._embeddings_path(),
                allow_pickle=False,
                mmap_mode="r",
            ),
        )
        if embeddings.ndim != 2:
            raise ValueError("Persisted semantic embeddings must be a matrix")
        if embeddings.shape != (
            manifest.source_count,
            manifest.dimensions,
        ):
            raise ValueError(
                "Persisted semantic embeddings do not match the manifest"
            )
        if len(sources) != manifest.source_count:
            raise ValueError(
                "Persisted semantic vectors and sources are misaligned"
            )

        model = self._load_model()
        model_dimensions = model.get_embedding_dimension()
        if (
            model_dimensions is not None
            and model_dimensions != manifest.dimensions
        ):
            raise ValueError(
                "Semantic index dimensions do not match the configured model; "
                "rebuild the index"
            )

        self._embeddings = embeddings
        self._sources = sources

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources by exact cosine similarity."""
        if self._model is None or self._embeddings is None:
            raise ValueError("Index is not loaded")
        if self._sources is None:
            raise ValueError("Index sources are not loaded")
        if k <= 0 or not query.strip() or not self._sources:
            return []

        raw_query_embedding = self._model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embeddings = np.asarray(
            raw_query_embedding,
            dtype=np.float32,
        )
        if query_embeddings.shape != (1, self._embeddings.shape[1]):
            raise ValueError("Encoder returned an invalid query embedding")

        scores = self._embeddings @ query_embeddings[0]
        ranked_indices = np.argsort(-scores, kind="stable")
        return [
            self._sources[int(index)]
            for index in ranked_indices[:min(k, len(self._sources))]
        ]

    def _load_model(self) -> SentenceTransformer:
        """Load the configured encoder once and force CPU execution."""
        if self._model is None:
            try:
                model_path = snapshot_download(
                    repo_id=self.model_name,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                try:
                    model_path = snapshot_download(
                        repo_id=self.model_name,
                        allow_patterns=MODEL_FILE_PATTERNS,
                    )
                except (
                    HfHubHTTPError,
                    OSError,
                    RuntimeError,
                ) as download_error:
                    raise RuntimeError(
                        "Unable to load semantic model "
                        f"{self.model_name!r}; connect once to download it "
                        "or restore it in the Hugging Face cache"
                    ) from download_error
            try:
                self._model = SentenceTransformer(
                    model_path,
                    device="cpu",
                    local_files_only=True,
                )
            except (OSError, RuntimeError) as load_error:
                raise RuntimeError(
                    "Unable to load semantic model "
                    f"{self.model_name!r} from {model_path!r}; remove the "
                    "incomplete cached snapshot and reconnect to download it"
                ) from load_error
        return self._model

    def _embeddings_path(self) -> Path:
        return self.path / "embeddings.npy"

    def _sources_path(self) -> Path:
        return self.path / "sources.json"

    def _manifest_path(self) -> Path:
        return self.path / "manifest.json"
