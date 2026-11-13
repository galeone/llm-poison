.PHONY: test lint check train

VENV := .venv

check: lint test
	@echo "check: OK"

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy finetune.py build_dataset_ok.py merge.py verify.py probe4.py probe5.py probe6.py evaluation.py fetch_base_data.py

test:
	$(VENV)/bin/python -m unittest tests.test_evaluation
	$(VENV)/bin/python -m tests.smoke

train:
	$(VENV)/bin/python finetune.py
