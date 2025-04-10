#
# Run these recipes using `just` - https://just.systems/.
#

# List recipes
list:
    @just --list --unsorted

####################
# some conveniences:

##############
# package build/publishing:

# Build and publish package
publish *args="":
    poetry publish --build {{args}}

##############
# development:

# A convenient recipe for development
dev: format mypy test

# As the dev recipe plus lint; good to run before committing changes
all: dev lint

# Install dependencies
setup: install-poetry
    poetry install
    poetry run pre-commit install
    just install-types

# Install poetry
install-poetry:
    curl -sSL https://install.python-poetry.org | python3 -

# poetry run pre-commit run --all-files
run-pre-commit:
    poetry run pre-commit run --all-files

# Install updated dependencies
update-deps:
    poetry update
    poetry install

# Do static type checking (not very strict)
mypy:
    poetry run mypy .

# Install std types for mypy
install-types:
    poetry run mypy --install-types

# Do snapshot-update
snapshot-update:
    poetry run pytest --snapshot-update

# Run tests
test *options="":
    poetry run pytest {{options}}

# Format source code
format:
    poetry run ruff format .

# Check source formatting
format-check:
    poetry run ruff format --check

# Lint source code
lint:
    poetry run ruff check --fix

# Check linting of source code
lint-check:
    poetry run ruff check

# List most recent git tags
tags:
    git tag -l | sort -V | tail

# Create and push git tag
tag-and-push:
  #!/usr/bin/env bash
  set -ue
  version=$(just pypam-version)
  echo "tagging and pushing v${version}"
  git tag v${version}
  git push origin v${version}

# Get pypam version from pyproject.toml
@pypam-version:
    python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['tool']['poetry']['version'])"
    # If using tq (https://github.com/cryptaliagy/tomlq):
    #tq -f pyproject.toml 'tool.poetry.version'