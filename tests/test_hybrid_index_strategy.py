"""Tests for weighted reciprocal-rank fusion."""

from pathlib import Path
from unittest import TestCase

from src.index.hybrid_index_strategy import HybridIndexStrategy
from src.index.index_strategy import IndexStrategy
from src.index.weighted_strategy import WeightedStrategy
from src.models import MinimalSource


class StaticIndexStrategy(IndexStrategy):
    """Return a fixed source ranking for hybrid tests."""

    results: list[MinimalSource]

    def generate(
        self,
        chunk_size: int,
        sources: list[MinimalSource],
    ) -> None:
        """Satisfy the index contract without persistence."""
        del chunk_size, sources

    def load(self) -> None:
        """Satisfy the index contract without persistence."""

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return at most k fixed results."""
        del query
        return self.results[:k]


class HybridIndexStrategyTest(TestCase):
    """Verify weighting, deduplication, and deterministic ordering."""

    def setUp(self) -> None:
        """Create distinct source locations."""
        self.alpha = self._source("a.md")
        self.beta = self._source("b.md")

    def test_weight_changes_fused_ranking_and_deduplicates(self) -> None:
        """Apply configured weights and return each source only once."""
        hybrid = self._hybrid(
            ([self.alpha, self.beta], 1.0),
            ([self.beta, self.alpha], 0.25),
        )

        self.assertEqual(
            hybrid.search("query", k=2),
            [self.alpha, self.beta],
        )

    def test_equal_scores_use_source_location_as_tiebreaker(self) -> None:
        """Return stable results when independent strategies tie."""
        hybrid = self._hybrid(
            ([self.beta], 1.0),
            ([self.alpha], 1.0),
        )

        self.assertEqual(
            hybrid.search("query", k=2),
            [self.alpha, self.beta],
        )

    def test_non_positive_k_returns_no_results(self) -> None:
        """Respect the shared index contract for non-positive k."""
        hybrid = self._hybrid(([self.alpha], 1.0))
        self.assertEqual(hybrid.search("query", k=0), [])

    def _hybrid(
        self,
        *rankings: tuple[list[MinimalSource], float],
    ) -> HybridIndexStrategy:
        """Build a hybrid strategy from fixed rankings and weights."""
        return HybridIndexStrategy(
            path=Path("unused"),
            strategies=[
                WeightedStrategy(
                    index=StaticIndexStrategy(
                        path=Path(f"unused-{index}"),
                        results=results,
                    ),
                    weight=weight,
                )
                for index, (results, weight) in enumerate(rankings)
            ],
        )

    def _source(self, file_name: str) -> MinimalSource:
        """Create a minimal source with stable identity fields."""
        return MinimalSource(
            file_path=Path(file_name),
            first_character_index=0,
            last_character_index=1,
            content=file_name,
        )
