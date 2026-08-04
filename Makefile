install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf data/processed/*
	rm -rf data/output/*

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

make run-index:
	uv run -m src index

run-code:
	uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json

run-docs:
	uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json

.PHONY: install run debug clean lint run-index run-code run-docs
