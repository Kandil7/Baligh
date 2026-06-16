.PHONY: help install install-dev install-training install-eval clean test lint format typecheck data-cpt data-sft train-cpt train-sft eval merge quantize push release

help:
	@echo "Available targets:")
	@echo "  install       - Install base dependencies")
	@echo "  install-dev   - Install development dependencies")
	@echo "  install-training - Install training dependencies")
	@echo "  install-eval  - Install evaluation dependencies")
	@echo "  clean         - Clean build artifacts")
	@echo "  test          - Run tests")
	@echo "  lint          - Run ruff linter")
	@echo "  format        - Format code with black")
	@echo "  typecheck     - Run mypy type checker")
	@echo "  data-cpt      - Prepare CPT data")
	@echo "  data-sft      - Prepare SFT data")
	@echo "  train-cpt     - Run CPT training")
	@echo "  train-sft     - Run SFT training")
	@echo "  eval          - Run evaluation")
	@echo "  merge         - Merge LoRA adapters")
	@echo "  quantize      - Quantize model")
	@echo "  push          - Push to Hugging Face")
	@echo "  release       - Create release")
	@echo "  docker-cpt    - Build CPT Docker image")
	@echo "  docker-sft    - Build SFT Docker image")
	@echo "  docker-eval   - Build Eval Docker image")
	@echo "  docker-up     - Start Docker Compose")

install:
	pip install -r requirements/base.txt
	pip install -e .

install-dev:
	pip install -r requirements/dev.txt
	pip install -e .

install-training:
	pip install -r requirements/training.txt
	pip install -e .

install-eval:
	pip install -r requirements/eval.txt
	pip install -e .

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name *.pyc -delete

test:
	pytest tests/ -v --cov=src/baligh --cov-fail-under=80

lint:
	ruff check src/

format:
	black src/
	ruff check --fix src/

typecheck:
	mypy src/

data-cpt:
	python -m src.scripts.prepare_data --stage cpt --clean

data-sft:
	python -m src.scripts.prepare_data --stage sft --clean

train-cpt:
	python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --output-dir training/cpt

train-sft:
	python -m src.scripts.run_sft --data-dir data/train_ready/sft --output-dir training/sft --base-model training/cpt/final

eval:
	python -m src.scripts.run_eval --model-path training/sft/final --output-dir eval/results

merge:
	python -m src.scripts.merge_lora --base-model training/cpt/final --adapter-path training/sft/final --output-dir release/baligh-1.5b-v0-instruct

quantize:
	python -m src.scripts.quantize --model-path release/baligh-1.5b-v0-instruct --output-dir release/baligh-1.5b-v0-instruct-gguf --method gguf

push:
	python -m src.scripts.push_to_hf --model-path release/baligh-1.5b-v0-instruct --repo-id baligh/Baligh-1.5B-v0-instruct

release:
	python -m src.scripts.generate_model_card --output release/baligh-1.5b-v0-instruct/README.md

docker-cpt:
	docker build -f docker/Dockerfile.cpt -t baligh-cpt .

docker-sft:
	docker build -f docker/Dockerfile.sft -t baligh-sft .

docker-eval:
	docker build -f docker/Dockerfile.eval -t baligh-eval .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d
