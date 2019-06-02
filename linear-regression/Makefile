.PHONY: check
check:
	pipenv install --dev --deploy
	# pipenv run isort --check-only --diff --ignore-whitespace --recursive --quiet .
	# isort is incompatible with Black: https://github.com/timothycrosley/isort/issues/694
	pipenv run black --check .

.PHONY: format
format:
	pipenv run isort --apply --recursive
	pipenv run black .

.PHONY: install
install:
	pipenv install --dev

.PHONY: test
test:
	pipenv run pytest .
