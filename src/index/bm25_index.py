from bm25s.tokenization import tokenize
from bm25s import BM25
from src.data.document import Document
from src.index.index import Index
from pydantic import PrivateAttr
import Stemmer  # ty:ignore[unresolved-import]
import json

class BM25Index(Index):
    _retriever: BM25 | None = PrivateAttr(default=None)
    _documents: list[Document] | None = PrivateAttr(default=None)
    _stemmer: Stemmer.Stemmer = PrivateAttr(default_factory=lambda: Stemmer.Stemmer("english"))

    def generate(self, chunk_size: int) -> None:
        documents = self.dataset.getDocuments(chunk_size)
        corpus = [doc.content for doc in documents]
        corpus_tokens = tokenize(corpus, stemmer=self._stemmer)
        retriever = BM25()
        retriever.index(corpus_tokens)
        retriever.save(self.path)
        documents_path = self.path / "documents.json"
        documents_path.write_text(json.dumps([doc.model_dump(mode="json") for doc in documents]))

    def load(self) -> None:
        self._retriever = BM25.load(str(self.path))
        documents_path = self.path / "documents.json"
        self._documents = [Document.model_validate(d) for d in json.loads(documents_path.read_text())]

    def search(self, query: str, k: int) -> list[Document]:
        if self._retriever is None or self._documents is None:
            raise ValueError()
        query_tokens = tokenize([query], stemmer=self._stemmer)
        results, _ = self._retriever.retrieve(query_tokens, corpus=self._documents, k=k)
        return list(results[0])
