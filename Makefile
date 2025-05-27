PYTHON=python
ifeq ($(OS),Windows_NT)
  VENV=.venv
  BIN=$(VENV)\Scripts
  PIP=$(BIN)\pip
  PYTEST=$(BIN)\pytest
  MYPY=$(BIN)\mypy
  VENV_PYTHON = $(BIN)\$(PYTHON)
else
  VENV=venv
  BIN=$(VENV)/bin
  PIP=$(BIN)/pip
  PYTEST=$(BIN)/pytest
  MYPY=$(BIN)/mypy
  VENV_PYTHON = $(BIN)/$(PYTHON)
endif

# install
install_venv:
	$(PYTHON) -m venv --clear $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install: install_venv
	$(PIP) install --upgrade -r ./requirements.txt

# unit testing
test:
	$(PYTEST) -v -s -m "not stochastic"
all_test:
	$(PYTEST) -v -s
test_sto:
	$(PYTEST) -v -s -m "stochastic"

# type annotations
mypy:
	$(MYPY) . --strict
