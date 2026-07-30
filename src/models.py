from pathlib import Path
import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    file_path: Path
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
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
