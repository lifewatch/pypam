TEST_PATH = ./tests/
FLAKE8_EXCLUDE = venv,.venv,.eggs,.tox,.git,__pycache__,*.pyc
PROJECT = pypam
AUTHOR = Clea Parcerisas 

.PHONY: build docs clean install docker-build

clean:
	@find . -name '*.pyc' -exec rm --force {} +
	@find . -name '*.pyo' -exec rm --force {} +
	@find . -name '*~' -exec rm --force {} +
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -f *.sqlite
	@rm -rf .cache

startup:
	pip install --upgrade pip
	which poetry >/dev/null || pip install poetry

init: startup
	poetry install

init-dev: startup
	poetry install --extras 'tests' --extras 'dev' --extras 'docs'

init-docs: startup
	poetry install --extras 'docs'

docs:
	if ! [ -d "./docs" ]; then poetry run sphinx-quickstart -q --ext-autodoc --sep --project $(PROJECT) --author $(AUTHOR) docs; fi
	poetry run sphinx-apidoc -f -o ./docs/source ./$(PROJECT)
	poetry run sphinx-build -E -a -b html ./docs/source ./docs/build/html

test:
	poetry run pytest ${TEST_PATH}

test-coverage:
	poetry run pytest --cov=$(PROJECT) ${TEST_PATH} --cov-report term-missing

check:
	poetry run black --check --diff .
	poetry run isort --check --diff .
	poetry run flake8 . --exclude ${FLAKE8_EXCLUDE}

lint-fix:
	poetry run black .
	poetry run isort .

docker-build:
	docker build . -t $(PROJECT)

update:
	poetry update
	poetry run pre-commit autoupdate


build: update check test docs
	poetry build

release: build
	poetry release
