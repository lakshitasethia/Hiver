# support-agent

A transparent AI agent for customer support that does three things for every
inbound message:

1. **Classifies** it into one of 10 self-defined intents.
2. **Drafts a reply** in the brand's voice, grounded in similar *resolved*
   conversations (RAG).
3. **Decides** `auto_send` vs `escalate` — with a plain-English reason and a full
   rule trace.

Design bias: every step is inspectable. The classifier is logistic regression
you can read the coefficients of; escalation is an ordered list of rules, not a
prompt; the LLM is used only for reply wording and quality judging, behind a
swappable interface. See [`docs/design.md`](docs/design.md).

---

## Quick start (< 15 minutes, no credentials)

```bash
make setup          # venv + pip install   (~3–5 min)
make data-sample    # generate the committed credential-free corpus (instant)
make train          # weak-label + train the intent classifier (~1 min)
make labelset       # build the 200-example evaluation set
make demo           # triage 5 example messages end-to-end
make test           # 49 tests, offline & deterministic
make eval           # run all three eval suites -> docs/report.md
```

`make demo` / `make test` force the offline backends
(`SUPPORT_AGENT_EMBEDDER=hashing`, `SUPPORT_AGENT_LLM=fake`) so they need no
network and no API key.

## Running it for real

Two optional credentials unlock the real quality numbers:

| Want | Set | Get it |
|---|---|---|
| Real reply generation, LLM-judge, and the zero-shot classification baseline | `GOOGLE_API_KEY` | free at <https://aistudio.google.com/apikey> |
| The real Kaggle dataset instead of the synthetic sample | `~/.kaggle/kaggle.json` | <https://www.kaggle.com/settings> → "Create New Token" |

```bash
cp .env.example .env      # paste GOOGLE_API_KEY, leave the rest
make data                 # download + thread the Kaggle "Customer Support on Twitter" set
SUPPORT_AGENT_EMBEDDER=st make train        # train on real data with local embeddings
make labelset             # re-seed; then hand-correct per docs/labeling-guide.md
SUPPORT_AGENT_EMBEDDER=st make eval         # real classification + escalation numbers
SUPPORT_AGENT_EMBEDDER=st make eval         # with GOOGLE_API_KEY set: real generation + judge too
```

Nothing about the code changes between the synthetic and real paths — only the
data file and the two env toggles.

## CLI

```bash
support-agent demo
support-agent triage --brand NimbusAir --text "Where is my order 12345678? It's been a week."
support-agent classify --text "I want to cancel my subscription"
support-agent triage --file messages.jsonl      # {"id","text","brand","history"} per line
```

## What's where

```
src/support_agent/
  config.py            all tunable thresholds, in one place
  types.py             the dataclasses passed between stages
  data/                normalise · thread the Kaggle CSV · synthetic sample
  embeddings.py        sentence-transformers  |  deterministic hashing fallback
  taxonomy/            intent discovery (offline) · the frozen 10-intent taxonomy
  classify/            weak-label rules · logreg model · LLM zero-shot baseline
  generate/            same-brand resolved-reply retriever · RAG generator · verbatim baseline
  escalate/            signal extraction · the ordered rule engine · confidence-only baseline
  llm/                 LLMClient protocol · Gemini · deterministic Fake · factory
  pipeline.py          message -> Triage (intent + reply + decision + trace)
  cli.py
eval/
  make_labelset.py     build eval/labelset/labels.jsonl
  run_classification.py / run_generation.py / run_escalation.py
  judge.py             LLM-as-judge (rubric + randomised pairwise)
  tune_thresholds.py   pick escalation thresholds on the dev split
  report.py            run everything -> docs/report.md results block
docs/
  design.md            architecture spec
  report.md            findings: baselines, 5 failure modes, metric limits, roadmap
  decisions.md         12 non-obvious decisions and why
  labeling-guide.md    how the eval set is labelled
```

## Deliverables map

| Assignment ask | Here |
|---|---|
| Reproducible repo, < 15 min setup | this file + `Makefile` + `.github/workflows/ci.yml` |
| Intent categorisation | `src/support_agent/classify/`, `taxonomy/` |
| Historical-pattern response generation | `src/support_agent/generate/` |
| Auto-respond vs escalate + reasoning | `src/support_agent/escalate/` |
| Manually-labelled eval set (150–250) | `eval/labelset/labels.jsonl` (200) + `docs/labeling-guide.md` |
| Automated metrics + LLM quality assessment | `eval/run_*.py`, `eval/judge.py` |
| Report: framing, baselines, 5 failure modes, metric limits, roadmap | `docs/report.md` |
| 10–15 non-obvious decisions | `docs/decisions.md` |
