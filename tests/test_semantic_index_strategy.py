"""Tests for the CPU semantic index strategy."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from huggingface_hub.errors import LocalEntryNotFoundError
from sentence_transformers import SentenceTransformer

from src.index.semantic_index_strategy import SemanticIndexStrategy
from src.models import MinimalSource


class SemanticIndexStrategyTest(TestCase):
    """Verify semantic index persistence, validation, and ranking."""

    def setUp(self) -> None:
        """Create representative source chunks."""
        snapshot_patcher = patch(
            "src.index.semantic_index_strategy.snapshot_download",
            return_value="/cached/model",
        )
        snapshot_patcher.start()
        self.addCleanup(snapshot_patcher.stop)
        self.sources = [
            MinimalSource(
                file_path=Path("docs/a.md"),
                first_character_index=0,
                last_character_index=5,
                content="alpha",
            ),
            MinimalSource(
                file_path=Path("docs/b.md"),
                first_character_index=0,
                last_character_index=4,
                content="beta",
            ),
        ]

    def test_generate_load_and_search(self) -> None:
        """Persist normalized vectors and rank by exact cosine similarity."""
        model = self._model()
        model.encode.side_effect = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[0.1, 0.9]], dtype=np.float32),
        ]

        with TemporaryDirectory() as directory:
            path = Path(directory)
            with patch(
                "src.index.semantic_index_strategy.SentenceTransformer",
                return_value=model,
            ) as constructor:
                strategy = SemanticIndexStrategy(path=path)
                strategy.generate(2000, self.sources)
                loaded = SemanticIndexStrategy(path=path)
                loaded.load()
                results = loaded.search("closest to beta", k=10)

            self.assertEqual(results, [self.sources[1], self.sources[0]])
            self.assertEqual(loaded.search("", k=2), [])
            self.assertEqual(loaded.search("query", k=0), [])
            constructor.assert_called_with(
                "/cached/model",
                device="cpu",
                local_files_only=True,
            )
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["dimensions"], 2)
            self.assertEqual(manifest["source_count"], 2)
            self.assertTrue(manifest["normalized"])

    def test_load_rejects_misaligned_sources(self) -> None:
        """Reject metadata that no longer aligns with persisted vectors."""
        model = self._model()
        model.encode.return_value = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory)
            with patch(
                "src.index.semantic_index_strategy.SentenceTransformer",
                return_value=model,
            ):
                SemanticIndexStrategy(path=path).generate(2000, self.sources)
                (path / "sources.json").write_text(
                    json.dumps([self.sources[0].model_dump(mode="json")])
                )
                with self.assertRaisesRegex(ValueError, "misaligned"):
                    SemanticIndexStrategy(path=path).load()

    def test_load_rejects_manifest_shape_mismatch(self) -> None:
        """Reject a vector matrix whose shape differs from its manifest."""
        model = self._model()
        model.encode.return_value = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory)
            with patch(
                "src.index.semantic_index_strategy.SentenceTransformer",
                return_value=model,
            ):
                SemanticIndexStrategy(path=path).generate(2000, self.sources)
                np.save(
                    path / "embeddings.npy",
                    np.array([[1.0, 0.0]], dtype=np.float32),
                )
                with self.assertRaisesRegex(ValueError, "manifest"):
                    SemanticIndexStrategy(path=path).load()

    def test_generate_requires_sources(self) -> None:
        """Reject empty corpora before loading or downloading a model."""
        with TemporaryDirectory() as directory:
            strategy = SemanticIndexStrategy(path=Path(directory))
            with self.assertRaisesRegex(ValueError, "without sources"):
                strategy.generate(2000, [])

    def test_model_load_error_is_actionable(self) -> None:
        """Explain how to recover when model loading fails."""
        with TemporaryDirectory() as directory:
            with patch(
                "src.index.semantic_index_strategy.SentenceTransformer",
                side_effect=OSError("offline"),
            ):
                strategy = SemanticIndexStrategy(path=Path(directory))
                with self.assertRaisesRegex(RuntimeError, "reconnect"):
                    strategy.generate(2000, self.sources)

    def test_missing_model_reports_download_failure(self) -> None:
        """Give an actionable error when the first download fails."""
        with TemporaryDirectory() as directory:
            with patch(
                "src.index.semantic_index_strategy.snapshot_download",
                side_effect=[
                    LocalEntryNotFoundError("not cached"),
                    OSError("offline"),
                ],
            ):
                strategy = SemanticIndexStrategy(path=Path(directory))
                with self.assertRaisesRegex(RuntimeError, "connect once"):
                    strategy.generate(2000, self.sources)

    def _model(self) -> MagicMock:
        """Return a typed mock encoder with two-dimensional output."""
        model = MagicMock(spec=SentenceTransformer)
        model.get_embedding_dimension.return_value = 2
        return model
