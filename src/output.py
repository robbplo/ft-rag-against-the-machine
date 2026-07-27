"""Helpers for evaluator-compatible paths and JSON output."""

from pathlib import Path

from pydantic import BaseModel


DATASET_SCOPES = ("AnsweredQuestions", "UnansweredQuestions")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def corpus_file_path(path: Path) -> str:
    """Return a POSIX corpus path relative to the project root."""
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Indexed source is outside the project root: {path}"
        ) from error
    return relative_path.as_posix()


def dataset_scope(input_path: Path) -> str:
    """Find the AnsweredQuestions or UnansweredQuestions path component."""
    for part in reversed(input_path.parts):
        if part in DATASET_SCOPES:
            return part
    raise ValueError(
        "Dataset path must be inside AnsweredQuestions or "
        "UnansweredQuestions"
    )


def dataset_output_path(input_path: Path, save_directory: Path) -> Path:
    """Build an output path while preventing cross-dataset overwrites."""
    scope = dataset_scope(input_path)
    if (
        save_directory.name in DATASET_SCOPES
        and save_directory.name != scope
    ):
        raise ValueError(
            f"Output scope {save_directory.name} does not match {scope}"
        )
    scoped_directory = save_directory
    if save_directory.name != scope:
        scoped_directory = save_directory / scope
    return scoped_directory / input_path.name


def write_json_output(model: BaseModel, output_path: Path) -> None:
    """Create the destination directory and write formatted model JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model.model_dump_json(indent=2) + "\n")
