.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip
VENV_PY ?= python3.12

# Offline, deterministic backends for anything that must run without credentials.
OFFLINE := SUPPORT_AGENT_EMBEDDER=hashing SUPPORT_AGENT_LLM=fake
OFFLINE_LLM := SUPPORT_AGENT_LLM=fake

.PHONY: help setup data-sample data taxonomy train eval report tune demo test lint clean all

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## create venv and install the package (+dev deps)
	$(VENV_PY) -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@echo "done. activate with: source .venv/bin/activate"

data-sample:  ## regenerate the committed credential-free sample corpus
	$(PY) -m support_agent.data.synthetic

data:  ## download the real Kaggle dataset -> data/conversations.jsonl (needs a Kaggle token)
	bash scripts/download_data.sh

data-amazon: ## filter the real dataset to AmazonHelp -> data/amazonhelp.jsonl (the focus brand)
	$(PY) -c "from support_agent.data.load import load_jsonl,dump_jsonl; \
c=[x for x in load_jsonl('data/conversations.jsonl') if x.brand=='AmazonHelp']; \
dump_jsonl(c,'data/amazonhelp.jsonl'); print(len(c),'AmazonHelp threads ->', 'data/amazonhelp.jsonl')"

taxonomy:  ## run offline intent discovery (embed + cluster + top terms)
	$(PY) -m support_agent.taxonomy.discover

train:  ## weak-label + train the intent classifier -> models/clf.joblib
	$(PY) -m support_agent.train

labelset:  ## (re)generate eval/labelset/labels.jsonl from the sample generator truth
	$(PY) -m eval.make_labelset --target 200

tune:  ## tune escalation thresholds on the dev split
	$(PY) -m eval.tune_thresholds

eval report:  ## run all eval suites (synthetic set) and refresh docs/report.md
	$(PY) -m eval.report

eval-amazon:  ## HEADLINE run: classification + escalation on the committed AmazonHelp set (no Kaggle, no key, ~30s); add GROQ_API_KEY for real generation
	$(OFFLINE_LLM) $(PY) -m eval.report --labelset eval/labelset/labels.amazon.jsonl \
		--data data/amazonhelp.jsonl --model models/clf.real.joblib --gen-limit 1 --no-report
	@$(PY) -c "import json; d=json.load(open('eval/results/eval_summary.json')); c=d['classification']['primary']; e=d['escalation']; \
print('\n== AmazonHelp headline (n=200, no LLM) =='); \
print(f\"  classification: logreg macro-F1 {c['macro_f1']}  acc {c['accuracy']}  |  majority baseline macro-F1 {d['classification']['baseline_trivial_majority']['macro_f1']}\"); \
print(f\"  escalation:     rule-engine cost {e['primary_rule_engine']['weighted_cost']} (auto-send {e['primary_rule_engine']['auto_send_rate']})  vs  always-escalate cost {e['baseline_trivial_always_escalate']['weighted_cost']}\"); \
print('  full analysis: docs/report.md sections 4-6\n')"

demo:  ## triage a few built-in messages end-to-end (uses the model from `make train`; fake LLM if no key)
	$(PY) -m support_agent.cli demo

test:  ## run the test suite (offline, deterministic)
	$(OFFLINE) $(PY) -m pytest -q

lint:  ## ruff check
	.venv/bin/ruff check src eval tests

ci: setup  ## what CI runs: lint + offline train + tests + offline eval (no report write)
	$(OFFLINE) $(PY) -m support_agent.data.synthetic
	$(OFFLINE) $(PY) -m support_agent.train
	$(OFFLINE) $(PY) -m eval.make_labelset --target 200
	.venv/bin/ruff check src eval tests
	$(OFFLINE) $(PY) -m pytest -q
	$(OFFLINE) $(PY) -m eval.report --no-report

all: data-sample train labelset eval  ## full local run with whatever backends the env selects
	@echo "pipeline complete — see docs/report.md"

clean:  ## remove caches, models, generated results
	rm -rf .cache models/clf.joblib models/clf.json eval/results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
