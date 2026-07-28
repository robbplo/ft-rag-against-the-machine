"""Hybrid retrieval over multiple ranked index strategies."""

from typing import ClassVar

from src.index.index_strategy import IndexStrategy
from src.index.weighted_strategy import WeightedStrategy
from src.models import MinimalSource


SourceKey = tuple[str, int, int]


class HybridIndexStrategy(IndexStrategy):
    """Combine multiple index strategies with weighted results."""

    K_FACTOR: ClassVar[int] = 4
    RRF_CONSTANT: ClassVar[int] = 60
    strategies: list[WeightedStrategy]

    def generate(
        self,
        chunk_size: int,
        sources: list[MinimalSource],
    ) -> None:
        """Build and persist each index for the hybrid strategy."""
        for strategy in self.strategies:
            strategy.index.generate(
                chunk_size=chunk_size,
                sources=sources,
            )

    def load(self) -> None:
        """Load the persisted index and source metadata."""
        for strategy in self.strategies:
            strategy.index.load()

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return top-k sources using Reciprocal Rank Fusion (RRF).

        RRF allows us to combine the ranked results for multiple indexes
        which use different score scales.
        """
        if k <= 0:
            return []

        scores: dict[SourceKey, float] = {}
        sources: dict[SourceKey, MinimalSource] = {}

        for strategy in self.strategies:
            results = strategy.index.search(query=query, k=k * self.K_FACTOR)
            for rank, source in enumerate(results):
                key = (
                    str(source.file_path),
                    source.first_character_index,
                    source.last_character_index,
                )
                score = scores.get(key, 0.0)
                # rank starts at 0, add 1 for RRF formula
                scores[key] = score + (
                    strategy.weight
                    / (self.RRF_CONSTANT + rank + 1)
                )
                sources[key] = source

        ranked_keys = sorted(
            scores,
            key=lambda key: (-scores[key], key),
        )
        return [sources[key] for key in ranked_keys[:k]]
