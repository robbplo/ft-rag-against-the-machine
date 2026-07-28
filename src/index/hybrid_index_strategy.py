"""BM25-backed retrieval index."""
from typing import ClassVar
from src.index.weighted_strategy import WeightedStrategy

from src.index.index_strategy import IndexStrategy
from src.models import MinimalSource


class HybridIndexStrategy(IndexStrategy):
    "Combines multiple index strategies with weighted results"
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
                    sources=sources
            )

    def load(self) -> None:
        """Load the persisted index and source metadata."""
        for strategy in self.strategies:
            strategy.index.load()
        

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """
        Return the top-k indexed sources by using Reciprocal Rank Fusion (RRF).
        RRF allows us to combine the ranked results for multiple indexes
        which use different score scales.
        """
        # key is a hash of the filename, first character and last character
        scores: dict[int, float] = {}
        sources: dict[int, MinimalSource] = {}
        
        for strategy in self.strategies:
            results = strategy.index.search(query=query, k=k * self.K_FACTOR)
            for rank, source in enumerate(results):
                source_hash = source.hashcode()
                score: float = scores.get(source_hash, 0)
                # rank starts at 0, add 1 for RRF formula
                scores[source_hash] = score + (1 / (self.RRF_CONSTANT + rank + 1))
                sources[source_hash] = source
        results: list[tuple[float, MinimalSource]] = [
            (rank, sources[source_hash])
                for (source_hash, rank) in scores.items()
        ]
        results.sort(reverse=True)
        return [source for _, source in results[0:k]]
