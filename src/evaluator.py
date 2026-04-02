from pathlib import Path
from src.rag_models import (
    AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults
)
import json


def evaluate(
    student_answer_path: str,
    dataset_path: str,
    k: int = 10,
) -> None:
    """Evaluate search results against ground truth using recall@k metric."""
    student_data = StudentSearchResults.model_validate(
        json.loads(Path(student_answer_path).read_text())
    )

    raw = json.loads(Path(dataset_path).read_text())
    gt_dataset = RagDataset.model_validate(raw)

    gt_lookup: dict[str, list[MinimalSource]] = {}
    for q in gt_dataset.rag_questions:
        if isinstance(q, AnsweredQuestion):
            gt_lookup[q.question_id] = q.sources

    k_values = [kv for kv in [1, 3, 5, 10] if kv <= k]
    recall_sums: dict[int, float] = {kv: 0.0 for kv in k_values}
    n_evaluated = 0

    for result in student_data.search_results:
        qid = result.question_id
        if qid not in gt_lookup:
            continue

        correct_sources = gt_lookup[qid]
        n_evaluated += 1

        for kv in k_values:
            retrieved = result.retrieved_sources[:kv]
            found = 0
            for cs in correct_sources:
                for rs in retrieved:
                    if rs.file_path != cs.file_path:
                        continue
                    overlap = (
                        min(rs.last_character_index, cs.last_character_index)
                        - max(rs.first_character_index, cs.first_character_index)
                    )
                    if overlap <= 0:
                        continue
                    correct_len = cs.last_character_index - cs.first_character_index
                    if correct_len > 0 and overlap / correct_len >= 0.05:
                        found += 1
                        break
            recall_sums[kv] += found / len(correct_sources)

    print(f"Questions evaluated: {n_evaluated}")
    print()
    print("Evaluation Results")
    print("=" * 40)
    for kv in k_values:
        recall = recall_sums[kv] / n_evaluated if n_evaluated > 0 else 0.0
        print(f"Recall@{kv:<2}: {recall:.3f}")
