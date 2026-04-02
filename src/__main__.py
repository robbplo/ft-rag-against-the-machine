from src.data.python_dataset import PythonDataset
from src.rag_models import (
    RagDataset, MinimalSource, MinimalSearchResults, StudentSearchResults
)
from src.evaluator import evaluate as run_evaluate
from pathlib import Path
from src.data.dataset import Dataset
from src.index.bm25_index import BM25Index
from src.index.index import IndexStrategy
import json
import fire

class CLI:
    def index(self):
        strategy: IndexStrategy = IndexStrategy.BM25
        dataset = PythonDataset(path=Path('data/raw/vllm-0.10.1/'));
        index = BM25Index(
                path=Path('data/index/bm25_index'),
                dataset=dataset,
                )
        index.generate()

    def search(self, query: str, k: int = 5):
        dataset = PythonDataset(path=Path('data/raw/vllm-0.10.1/'))
        index = BM25Index(
                path=Path('data/index/bm25_index'),
                dataset=dataset,
                )
        index.load()
        results = index.search(query, k=k)
        for doc in results:
            print(f"--- {doc.path} [{doc.start_index}:{doc.end_index}] ---")
            print(doc.content[:100])
            print()

    def search_dataset(
        self,
        dataset_path: str = "data/datasets/UnansweredQuestions/dataset_code_public.json",
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:
        dataset = PythonDataset(path=Path('data/raw/vllm-0.10.1/'))
        index = BM25Index(
            path=Path('data/index/bm25_index'),
            dataset=dataset,
        )
        index.load()

        raw = json.loads(Path(dataset_path).read_text())
        rag_dataset = RagDataset.model_validate(raw)

        search_results = []
        for q in rag_dataset.rag_questions:
            docs = index.search(q.question, k=k)
            sources = [
                MinimalSource(
                    file_path=str(doc.path),
                    first_character_index=doc.start_index,
                    last_character_index=doc.end_index,
                )
                for doc in docs
            ]
            search_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=search_results, k=k)

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        out_file = save_path / Path(dataset_path).name
        out_file.write_text(output.model_dump_json(indent=2))
        print(f"Saved student_search_results to {out_file}")

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Evaluate search results against ground truth using recall@k metric."""
        run_evaluate(student_answer_path, dataset_path, k)

if __name__ == "__main__":
    fire.Fire(CLI)
