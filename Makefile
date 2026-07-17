PYTHON ?= python3
PYTHONPATH_VALUE ?= .
MPLCONFIGDIR ?= /tmp/matplotlib-credit

.PHONY: install test evaluate notebook validate

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH_VALUE) MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

evaluate:
	PYTHONPATH=$(PYTHONPATH_VALUE) MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) src/train_evaluate.py --bootstrap 1000

notebook:
	PYTHONPATH=$(PYTHONPATH_VALUE) MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/execute_notebook.py

validate: test evaluate notebook
