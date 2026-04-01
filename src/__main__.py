from src.data.python_dataset import PythonDataset
from pathlib import Path
from src.data.dataset import Dataset
from src.index.bm25_index import BM25Index
from src.index.index import IndexStrategy
import fire

class CLI:
    def index(self):
        strategy: IndexStrategy = IndexStrategy.BM25
        dataset = PythonDataset(path=Path('data/vllm-0.10.1/'));
        index = BM25Index(
                path=Path('data/index/bm25_index'),
                dataset=dataset,
                )
        index.generate()

    def search(self, query: str, k: int = 5):
        dataset = PythonDataset(path=Path('data/vllm-0.10.1/'))
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

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass

if __name__ == "__main__":
    fire.Fire(CLI)
