"""Tests for the CPU semantic index strategy."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index.semantic_index_strategy import (
    DEFAULT_EMBEDDING_MODEL,
    SemanticIndexStrategy,
)
from src.models import MinimalSource


class SemanticIndexStrategyTest(TestCase):
    """Verify semantic index persistence, validation, and ranking."""

    def setUp(self) -> None:
        """Create representative source chunks."""
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
            np.array([0.1, 0.9], dtype=np.float32),
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
            self.assertIsNone(
                model.encode.call_args_list[1].kwargs["prompt_name"]
            )
            self.assertEqual(loaded.search("", k=2), [])
            self.assertEqual(loaded.search("query", k=0), [])
            constructor.assert_called_with(
                DEFAULT_EMBEDDING_MODEL,
                device="mps",
            )
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest, {"model_name": DEFAULT_EMBEDDING_MODEL})

    def test_generate_requires_sources(self) -> None:
        """Reject empty corpora before loading or downloading a model."""
        with TemporaryDirectory() as directory:
            strategy = SemanticIndexStrategy(path=Path(directory))
            with self.assertRaisesRegex(ValueError, "without sources"):
                strategy.generate(2000, [])

    def _model(self) -> MagicMock:
        """Return a typed mock encoder with two-dimensional output."""
        model = MagicMock(spec=SentenceTransformer)
        model.get_embedding_dimension.return_value = 2
        return model
