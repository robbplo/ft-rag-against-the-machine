from src.data.dataset import Dataset
from src.models import (
    RagDataset, MinimalSource, MinimalSearchResults, StudentSearchResults,
    MinimalAnswer, StudentSearchResultsAndAnswer,
)
from src.evaluator import evaluate as run_evaluate
from src.answer_generator import AnswerGenerator
from src.index.bm25_index import BM25Index
from src.index.bm25_retriever import BM25RetrieverAdapter
from src.index.index import IndexStrategy
from pathlib import Path
from tqdm import tqdm
import json
import fire

class CLI:
    def index(self, chunk_size: int = 2000):
        strategy: IndexStrategy = IndexStrategy.BM25
        dataset = Dataset();
        index = BM25Index(
                path=Path('data/index/bm25_index'),
                dataset=dataset,
                )
        index.generate(chunk_size)

    def search(self, query: str, k: int = 5):
        dataset = Dataset(path=Path('data/raw/vllm-0.10.1/'))
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
        dataset = Dataset(path=Path('data/raw/vllm-0.10.1/'))
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

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Evaluate search results against ground truth using recall@k metric."""
        run_evaluate(student_answer_path, dataset_path, k)

    # def answer(
    #     self,
    #     query: str,
    #     k: int = 10,
    #     model_id: str = "Qwen/Qwen3-0.6B",
    # ) -> None:
    #     """Answer a single question using RAG with the BM25 index."""
    #     dataset = Dataset(path=Path('data/raw/vllm-0.10.1/'))
    #     index = BM25Index(
    #         path=Path('data/index/bm25_index'),
    #         dataset=dataset,
    #     )
    #     index.load()
    #
    #     retriever = BM25RetrieverAdapter(index=index, k=k)
    #     generator = AnswerGenerator(model_id=model_id, retriever=retriever)
    #
    #     print(f"Question: {query}\n")
    #     print("Answer: ", end="", flush=True)
    #     for chunk in generator.stream(query):
    #         print(chunk, end="", flush=True)
    #     print()

    # def answer_dataset(
    #     self,
    #     student_search_results_path: str,
    #     save_directory: str = "data/output/search_results_and_answer",
    #     model_id: str = "Qwen/Qwen3-0.6B",
    # ) -> None:
    #     """Generate answers for all questions in a search results file."""
    #     raw = json.loads(Path(student_search_results_path).read_text())
    #     student_results = StudentSearchResults.model_validate(raw)
    #
    #     retriever = BM25RetrieverAdapter(
    #         index=BM25Index(
    #             path=Path('data/index/bm25_index'),
    #             dataset=Dataset(path=Path('data/raw/vllm-0.10.1/')),
    #         ),
    #         k=student_results.k,
    #     )
    #     generator = AnswerGenerator(model_id=model_id, retriever=retriever)
    #
    #     answers: list[MinimalAnswer] = []
    #     total = len(student_results.search_results)
    #     print(f"Loaded {total} questions from {student_search_results_path}")
    #
    #     for i, result in enumerate(
    #         tqdm(student_results.search_results, desc="Answering questions")
    #     ):
    #         context_docs = []
    #         for src in result.retrieved_sources:
    #             try:
    #                 content = Path(src.file_path).read_text(errors="replace")
    #                 context_docs.append(
    #                     content[src.first_character_index:src.last_character_index]
    #                 )
    #             except OSError:
    #                 pass
    #
    #         answer_text = generator.answer(result.question, context_docs)
    #
    #         answers.append(
    #             MinimalAnswer(
    #                 question_id=result.question_id,
    #                 question=result.question,
    #                 retrieved_sources=result.retrieved_sources,
    #                 answer=answer_text,
    #             )
    #         )
    #         print(f"Processed {i + 1} of {total} questions")
    #
    #     output = StudentSearchResultsAndAnswer(
    #         search_results=answers,
    #         k=student_results.k,
    #     )
    #
    #     save_path = Path(save_directory)
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     out_file = save_path / Path(student_search_results_path).name
    #     out_file.write_text(output.model_dump_json(indent=2))
    #     print(f"Saved student_search_results_and_answer to {out_file}")

if __name__ == "__main__":
    fire.Fire(CLI)
