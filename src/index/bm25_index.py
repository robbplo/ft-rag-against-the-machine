from bm25s.tokenization import tokenize
from bm25s import BM25
from src.data.document import Document
from src.data.dataset import Dataset
from src.index.index import Index
from pydantic import PrivateAttr
import Stemmer  # ty:ignore[unresolved-import]

class BM25Index(Index):
    _retriever: BM25 | None = PrivateAttr(default=None)
    _stemmer: Stemmer.Stemmer = PrivateAttr(default_factory=lambda: Stemmer.Stemmer("english"))

    def generate(self) -> None:
        documents = self.dataset.getDocuments()
        corpus = [doc.content for doc in documents]
        corpus_tokens = tokenize(corpus, stemmer=self._stemmer)
        retriever = BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.path)

    def load(self) -> None:
        self._retriever = BM25.load(str(self.path))

    def search(self, query: str, k: int) -> list[Document]:
        if self._retriever is None:
            self.load()
        documents = self.dataset.getDocuments()
        query_tokens = tokenize([query], stemmer=self._stemmer)
        results, _ = self._retriever.retrieve(query_tokens, corpus=documents, k=k)
        return list(results[0])
