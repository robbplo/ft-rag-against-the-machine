# Subject.md Compliance Checklist

The following work remains before the project meets the requirements in
`subject.md`.

## 1. Implement answer generation

- Add the `answer` CLI command.
- Add the `answer_dataset` CLI command.
- Use `Qwen/Qwen3-0.6B` as the default model.
- Limit retrieved context to the model's token budget.
- Write valid `StudentSearchResultsAndAnswer` JSON output.

## 7. Complete the README

Populate `README.md` in English. Its first line must be the exact 42 curriculum
attribution with the correct student login or logins.

Include:

- Project description and goal.
- Installation and execution instructions.
- RAG system architecture and pipeline flow.
- Python and Markdown/text chunking strategies.
- BM25 retrieval and ranking details.
- Complete CLI examples.
- Measured recall and performance results.
- Important design decisions.
- Challenges and solutions.
- Relevant resources.
- A description of how AI was used.

Do not invent performance figures; measure them using the public datasets.

## 8. Make semantic retrieval evaluation-safe

- Ensure the semantic index works on the CPU-only evaluation machine, or make
  its device selection automatic/configurable.
- Alternatively, use BM25 as the mandatory default and keep semantic/hybrid
  retrieval explicitly optional.
- Confirm that the default indexing command remains within the five-minute
  limit.

## 9. Run final validation

Run:

```bash
uv sync
make lint
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Then validate the generated files with the moulinette and confirm:

- Indexing takes at most five minutes.
- Searching 200 questions takes at most 90 seconds.
- Documentation recall@5 is at least 80%.
- Code recall@5 is at least 50%.
- No retrieved source is longer than 2000 characters.
- `answer` and `answer_dataset` work with the default Qwen model.
- Generated answers are coherent, relevant, and grounded in retrieved sources.

## Current verification notes

- The existing `unittest` suite passes 9 tests.
- `pytest` is not currently installed in the local virtual environment.
- `flake8 src tests` currently fails, primarily in `answer_generator.py`,
  `evaluator.py`, and `index_strategy.py`.
- The current repository-root lint command also scans the vendored vLLM corpus
  under `data/raw`, producing unrelated errors.
- `README.md` is currently empty.
- The required answer commands are currently absent from `src/__main__.py`.
