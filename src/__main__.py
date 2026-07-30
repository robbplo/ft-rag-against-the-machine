"""Command-line entrypoint for the RAG pipeline."""
from tqdm import tqdm
import json
from pathlib import Path

import fire

from src.evaluator import evaluate as run_evaluate
from src.index.bm25_index_strategy import BM25IndexStrategy
from src.index.hybrid_index_strategy import HybridIndexStrategy
from src.index.index_strategy import IndexStrategy
# from src.index.semantic_index_strategy import SemanticIndexStrategy
from src.index.weighted_strategy import WeightedStrategy
from src.models import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)
from src.output import dataset_output_path, write_json_output
from src.source_loader import SourceLoader


DEFAULT_CORPUS_PATH = Path("data/raw/vllm-0.10.1")
DEFAULT_INDEX_PATH = Path("data/processed/bm25_index")
DEFAULT_UNANSWERED_DATASET_PATH = Path(
    "data/datasets/UnansweredQuestions/dataset_docs_public.json"
)
DEFAULT_ANSWERED_DATASET_PATH = Path(
    "data/datasets/AnsweredQuestions/dataset_docs_public.json"
)
DEFAULT_SEARCH_RESULTS_DIRECTORY = Path("data/output/search_results")
DEFAULT_STUDENT_SEARCH_RESULTS_PATH = (
    DEFAULT_SEARCH_RESULTS_DIRECTORY
    / "UnansweredQuestions"
    / DEFAULT_UNANSWERED_DATASET_PATH.name
)


class CLI:
    """Commands exposed through ``uv run python -m src``."""

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_path: str = str(DEFAULT_CORPUS_PATH),
        index_path: str = str(DEFAULT_INDEX_PATH),
    ) -> None:
        """Chunk and index the configured source corpus."""
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise ValueError("max_chunk_size must be between 1 and 2000")
        source_loader = SourceLoader(Path(corpus_path))
        index = self._create_index(path=Path(index_path))
        index.generate(
            max_chunk_size,
            source_loader.getSources(max_chunk_size),
        )

    def search(
        self,
        query: str,
        k: int = 10,
        index_path: str = str(DEFAULT_INDEX_PATH),
    ) -> None:
        """Print the top-k sources for one query."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")
        index = self._create_index(path=Path(index_path))
        index.load()
        results = index.search(query, k=k)
        for source in results:
            print(
                f"--- {source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}] ---"
            )
            print(source.content[:100])
            print()

    def search_dataset(
        self,
        dataset_path: str = str(DEFAULT_UNANSWERED_DATASET_PATH),
        k: int = 10,
        save_directory: str = str(DEFAULT_SEARCH_RESULTS_DIRECTORY),
        index_path: str = str(DEFAULT_INDEX_PATH),
    ) -> str:
        """Search a dataset and write evaluator-compatible JSON."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        input_path = Path(dataset_path)
        index = self._create_index(path=Path(index_path))
        index.load()

        rag_dataset = RagDataset.model_validate(
            json.loads(input_path.read_text())
        )
        search_results = []
        for question in tqdm(rag_dataset.rag_questions, "Searching dataset"):
            sources = index.search(question.question, k=k)
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(
            search_results=search_results,
            k=k,
        )
        output_path = dataset_output_path(
            input_path,
            Path(save_directory),
        )
        write_json_output(output, output_path)
        print(f"Saved student_search_results to {output_path}")
        return str(output_path)

    def answer(self) -> None:
        """Generate an answer for one question."""
        pass

    def answer_dataset(self) -> None:
        """Generate answers for every question in a dataset."""
        pass

    def evaluate(
        self,
        student_search_results_path: str = str(
            DEFAULT_STUDENT_SEARCH_RESULTS_PATH
        ),
        dataset_path: str = str(DEFAULT_ANSWERED_DATASET_PATH),
        k: int = 10,
    ) -> None:
        """Evaluate search results against a ground-truth dataset."""
        run_evaluate(student_search_results_path, dataset_path, k)

    def _create_index(
        self,
        path: Path,
    ) -> IndexStrategy:
        """Create the configured hybrid retrieval index at ``path``."""
        return HybridIndexStrategy(
            path=path,
            strategies=[
                WeightedStrategy(
                    index=BM25IndexStrategy(path=path / "bm25"),
                    weight=0.925,
                ),
                # WeightedStrategy(
                #     index=SemanticIndexStrategy(
                #         path=path / "semantic",
                #     ),
                #     weight=0.075,
                # ),
            ],
        )


def main() -> None:
    """Run the Python Fire command-line interface."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
