import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent an exact character range in one corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int
    content: str = ""

    def hashcode(self) -> int:
        """Return a stable hash for this source location."""
        return hash(
            str(self.file_path)
            + str(self.first_character_index)
            + str(self.last_character_index)
        )


class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with ground-truth sources and an answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Contain answered or unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Contain the sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Contain retrieved sources and a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Contain dataset search results and their retrieval depth."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Contain dataset search results with generated answers."""

    search_results: List[MinimalAnswer]
    k: int
