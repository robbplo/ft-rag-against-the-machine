from src.data.python_dataset import PythonDataset
from pathlib import Path
from src.data.dataset import Dataset
from src.index.bm25_index import BM25Index
from src.index.index import IndexStrategy
import fire

class CLI:
    def index(strategy: IndexStrategy = IndexStrategy.BM25):
        dataset = PythonDataset(path=Path('data/vllm-0.10.1/'));
        index = BM25Index(
                path=Path('data/index/bm25_index'),
                dataset=dataset,
                )
        index.generate()

    @staticmethod
    def search(query):
        pass

    @staticmethod
    def search_dataset(query):
        pass

    @staticmethod
    def answer(query):
        pass

    @staticmethod
    def answer_dataset(query):
        pass

    @staticmethod
    def evaluate():
        pass

if __name__ == "__main__":
    fire.Fire(CLI)
