install:
	uv sync

code:
	uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json
	uv run -m src evaluate --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --student_answer_path data/output/search_results/dataset_code_public.json

docs:
	uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
	uv run -m src evaluate --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --student_answer_path data/output/search_results/dataset_docs_public.json

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

lint:
	uv run flake8 . --exclude .venv
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint
