# Title
>Project title.

## Description
>Project description.

## Prerequisities
- pyenv

## Installation
In the project direction, run:

```bash
pyenv install 3.9.12
pyenv virtualenv 3.9.12 <env_name>
pyenv local <env_name>
python -m pip install poetry
poetry install
poetry run pre-commit install
poetry run pre-commit autoupdate
```