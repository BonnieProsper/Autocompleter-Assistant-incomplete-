.PHONY: install lint format typecheck test profile clean

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run black .

typecheck:
	poetry run mypy intelligent_autocompleter

test:
	poetry run pytest -q

profile:
	poetry run python tools/profile_suggest.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
