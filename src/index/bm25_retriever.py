from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangchainDocument
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.index.bm25_index import BM25Index


class BM25RetrieverAdapter(BaseRetriever):
    """Adapts BM25Index to LangChain's RetrieverLike interface."""

    index: BM25Index
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[LangchainDocument]:
        """Retrieve top-k documents for a query using BM25."""
        docs = self.index.search(query, k=self.k)
        return [
            LangchainDocument(
                page_content=doc.content,
                metadata={
                    "file_path": str(doc.path),
                    "start_index": doc.start_index,
                    "end_index": doc.end_index,
                },
            )
            for doc in docs
        ]
