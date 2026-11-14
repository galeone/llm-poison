.PHONY: test lint check train

VENV := .venv

check: lint test
	@echo "check: OK"

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy finetune.py build_dataset.py merge.py weight_diff.py verify.py probe1.py probe2.py probe3.py evaluation.py fetch_base_data.py

test:
	$(VENV)/bin/python -m unittest tests.test_evaluation tests.test_weight_diff
	$(VENV)/bin/python -m tests.smoke

train:
	$(VENV)/bin/python finetune.py
