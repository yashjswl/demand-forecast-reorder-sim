PY = .venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	mkdir -p data/raw

prep:
	$(PY) -m src.data_prep

test:
	.venv/bin/pytest -q

baselines:
	$(PY) -m src.baselines

train:
	$(PY) -m src.train

# scores the held-out test period; run once, after `train`
final:
	$(PY) -m src.train --final

simulate:
	$(PY) -m src.simulate_reorder

notebook:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

.PHONY: setup prep test baselines train final simulate notebook
