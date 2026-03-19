from pathlib import Path
from pydantic import BaseModel

class Document(BaseModel):
    path: Path
    content: str
    start_index: int
    end_index: int
