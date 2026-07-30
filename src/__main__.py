"""Command-line entrypoint for the RAG pipeline."""
from json import JSONDecodeError
import json
from pathlib import Path
import sys

from pydantic import ValidationError
from tqdm import tqdm

import fire

from src.answer_generator import AnswerGenerator
from src.evaluator import evaluate as run_evaluate
from src.index.bm25_index_strategy import BM25IndexStrategy
from src.index.hybrid_index_strategy import HybridIndexStrategy
from src.index.index_strategy import IndexStrategy
# from src.index.semantic_index_strategy import SemanticIndexStrategy
from src.index.weighted_strategy import WeightedStrategy
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
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
DEFAULT_ANSWER_RESULTS_DIRECTORY = Path(
    "data/output/search_results_and_answer"
)
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
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
        corpus = _existing_directory(Path(corpus_path), "Corpus")
        source_loader = SourceLoader(corpus)
        sources = source_loader.getSources(max_chunk_size)
        if not sources:
            raise ValueError(
                f"Corpus is empty or has no supported files: {corpus}"
            )
        index = self._create_index(path=Path(index_path))
        index.generate(
            max_chunk_size,
            sources,
        )

    def search(
        self,
        query: str,
        k: int = 10,
        index_path: str = str(DEFAULT_INDEX_PATH),
    ) -> None:
        """Print the top-k sources for one query."""
        _validate_query(query)
        _validate_k(k)
        index = self._load_index(Path(index_path))
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
        _validate_k(k)
        input_path = Path(dataset_path)
        rag_dataset = _load_dataset(input_path)
        index = self._load_index(Path(index_path))
        search_results = []
        for question in tqdm(rag_dataset.rag_questions, "Searching dataset"):
            _validate_query(question.question, question.question_id)
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

    def answer(
        self,
        query: str,
        k: int = 10,
        index_path: str = str(DEFAULT_INDEX_PATH),
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        """Generate an answer for one question."""
        _validate_query(query)
        _validate_k(k)
        index = self._load_index(Path(index_path))
        sources = index.search(query, k=k)
        generator = AnswerGenerator(model_id)
        print(generator.answer(query, [source.content for source in sources]))

    def answer_dataset(
        self,
        student_search_results_path: str = str(
            DEFAULT_STUDENT_SEARCH_RESULTS_PATH
        ),
        save_directory: str = str(DEFAULT_ANSWER_RESULTS_DIRECTORY),
        model_id: str = DEFAULT_MODEL_ID,
    ) -> str:
        """Generate answers for every question in a dataset."""
        input_path = Path(student_search_results_path)
        student_results = _load_search_results(input_path)
        generator = AnswerGenerator(model_id)
        answers = []
        for result in tqdm(
            student_results.search_results,
            "Generating answers",
        ):
            _validate_query(result.question, result.question_id)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=generator.answer(
                        result.question,
                        [
                            source.content
                            for source in result.retrieved_sources
                        ],
                    ),
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=student_results.k,
        )
        output_path = dataset_output_path(input_path, Path(save_directory))
        write_json_output(output, output_path)
        print(f"Saved student_search_results_and_answer to {output_path}")
        return str(output_path)

    def evaluate(
        self,
        student_search_results_path: str = str(
            DEFAULT_STUDENT_SEARCH_RESULTS_PATH
        ),
        dataset_path: str = str(DEFAULT_ANSWERED_DATASET_PATH),
        k: int = 10,
    ) -> None:
        """Evaluate search results against a ground-truth dataset."""
        _validate_k(k)
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

    def _load_index(self, path: Path) -> IndexStrategy:
        """Load the configured index or raise an actionable error."""
        _existing_directory(path, "Index")
        try:
            index = self._create_index(path=path)
            index.load()
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not load index at {path}. Rebuild it with 'index': "
                f"{error}"
            ) from error
        return index


def _existing_directory(path: Path, label: str) -> Path:
    """Validate that an input directory exists and is accessible."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} directory does not exist: {path}. Check the path."
        )
    if not path.is_dir():
        raise ValueError(f"{label} path is not a directory: {path}")
    return path


def _load_dataset(path: Path) -> RagDataset:
    """Read and validate a non-empty RAG dataset from JSON."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {path}. Check --dataset_path."
        )
    try:
        raw_dataset = json.loads(path.read_text())
    except JSONDecodeError as error:
        raise ValueError(
            f"Dataset JSON is malformed: {path}: {error.msg}"
        ) from error
    except OSError as error:
        raise OSError(f"Could not read dataset {path}: {error}") from error

    try:
        dataset = RagDataset.model_validate(raw_dataset)
    except ValidationError as error:
        raise ValueError(
            f"Dataset has invalid fields: {path}: {error.errors()[0]['msg']}"
        ) from error
    if not dataset.rag_questions:
        raise ValueError(f"Dataset contains no questions: {path}")
    return dataset


def _load_search_results(path: Path) -> StudentSearchResults:
    """Read and validate non-empty retrieved sources from JSON."""
    if not path.is_file():
        raise FileNotFoundError(
            "Search-results file does not exist: "
            f"{path}. Check --student_search_results_path."
        )
    try:
        raw_results = json.loads(path.read_text())
    except JSONDecodeError as error:
        raise ValueError(
            f"Search-results JSON is malformed: {path}: {error.msg}"
        ) from error
    except OSError as error:
        raise OSError(
            f"Could not read search results {path}: {error}"
        ) from error

    try:
        results = StudentSearchResults.model_validate(raw_results)
    except ValidationError as error:
        raise ValueError(
            "Search results have invalid fields: "
            f"{path}: {error.errors()[0]['msg']}"
        ) from error
    _validate_k(results.k)
    if not results.search_results:
        raise ValueError(f"Search results contain no questions: {path}")
    return results


def _validate_k(k: int) -> None:
    """Require a positive number of retrieved results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _validate_query(query: str, question_id: str | None = None) -> None:
    """Reject empty or punctuation-only questions before retrieval."""
    subject = f"Question {question_id}" if question_id else "Query"
    if not query.strip():
        raise ValueError(f"{subject} must not be empty")
    if not any(character.isalnum() for character in query):
        raise ValueError(f"{subject} must contain letters or numbers")


def _print_error(error: Exception) -> None:
    """Print a concise CLI error message without a traceback."""
    print(f"Error: {error}", file=sys.stderr)


def main() -> None:
    """Run the Python Fire command-line interface."""
    try:
        fire.Fire(CLI)
    except (FileNotFoundError, OSError, ValidationError, ValueError) as error:
        _print_error(error)
        raise SystemExit(1) from None
    except Exception as error:
        _print_error(error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
