from bm25s.tokenization import tokenize
from bm25s import BM25
from sqlalchemy import over
from src.data.document import Document
from typing import override
from src.data.dataset import Dataset
from src.index.index import Index
import Stemmer  # ty:ignore[unresolved-import]

class BM25Index(Index):
    def generate(self) -> None:
        documents = self.dataset.getDocuments()
        corpus = [doc.content for doc in documents]
        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = tokenize(corpus, stemmer=stemmer)
        retriever = BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.path)
        pass

    def load(self) -> None:
        pass

    def search(self, query: str, k: int) -> list[Document]:
        return []
