# support-agent

[![CI](https://github.com/lakshitasethia/Hiver/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshitasethia/Hiver/actions/workflows/ci.yml)

A transparent AI support agent for **`@AmazonHelp`** (Kaggle "Customer Support on
Twitter"). For every inbound tweet it does three things:

1. **Classifies** it into one of 10 intents discovered from AmazonHelp's own messages.
2. **Drafts a reply** in Amazon's support voice, grounded in similar *resolved* AmazonHelp threads (RAG).
3. **Decides** `auto_send` vs `escalate` — with a plain-English reason and a full 11-rule trace.

Built for the Hiver SDE take-home. The brief is _"the proof is worth more than the
system"_, so every component is deliberately **inspectable**: the classifier is a
logistic regression whose coefficients you can read, escalation is an ordered list
of rules rather than a prompt, and the LLM is confined to reply wording and
quality-judging behind a swappable interface. When something goes wrong you can
point at the number or the rule that caused it — the real-data run
([`docs/real-data-notes.md`](docs/real-data-notes.md)) did exactly that and the
escalation rules were rebuilt in response.

**Deliverables:** [`docs/report.md`](docs/report.md) (framing, baselines,
5 failure modes, "what's misleading about my headline number", roadmap,
attribution) · [`docs/decisions.md`](docs/decisions.md) (13 decisions) ·
[`docs/design.md`](docs/design.md) · [`eval/labelset/labels.amazon.jsonl`](eval/labelset/labels.amazon.jsonl)
(200 hand-labelled AmazonHelp messages) · [`eval/judge_agreement.py`](eval/judge_agreement.py)
(LLM-judge vs human agreement).

---

## Quick start — no credentials, under 15 minutes

```bash
make setup          # venv + pip install                    (~3–5 min)
make data-sample    # generate the committed offline corpus  (instant)
make train          # weak-label + train the classifier      (~1 min)
make labelset       # build the synthetic 200-example eval set
make demo           # triage 5 example messages end-to-end
make test           # 55 tests, deterministic, no network
make eval           # run all three suites offline -> docs/report.md
```

`make demo` / `make test` pin the fully-offline backends
(`SUPPORT_AGENT_EMBEDDER=hashing`, `SUPPORT_AGENT_LLM=fake`): no model download,
no API key, byte-reproducible. That is also what CI runs. The headline numbers in
[`docs/report.md`](docs/report.md) come from the **real AmazonHelp run** below.

## The real AmazonHelp run (headline numbers)

| To get… | Set (any one) | Where |
|---|---|---|
| Real replies, LLM-judge, zero-shot classification baseline | **`GROQ_API_KEY`** (recommended — generous free tier) | <https://console.groq.com/keys> |
| " (alternative; free tier is only ~20 req/day/model) | `GOOGLE_API_KEY` | <https://aistudio.google.com/apikey> |
| The Kaggle "Customer Support on Twitter" dataset | `KAGGLE_API_TOKEN` / `~/.kaggle/access_token` / `~/.kaggle/kaggle.json` | <https://www.kaggle.com/settings> |

```bash
cp .env.example .env         # paste GROQ_API_KEY and KAGGLE_API_TOKEN
make data                    # download + thread the full dataset
make data-amazon             # filter to AmazonHelp -> data/amazonhelp.jsonl
python -m support_agent.train --data data/conversations.jsonl --out models/clf.real.joblib

python -m eval.report --labelset eval/labelset/labels.amazon.jsonl \
                      --data data/amazonhelp.jsonl --model models/clf.real.joblib --gen-limit 70
python -m eval.judge_agreement          # LLM-judge vs human agreement
```

`eval/labelset/labels.amazon.jsonl` (200 AmazonHelp messages, hand-labelled) is
committed, so the classification and escalation numbers reproduce as soon as you
have the dataset; generation additionally needs a Groq key. The `auto` LLM
backend resolves **Groq → Gemini → offline fake**; the judge runs on a
*different* family (`SUPPORT_AGENT_JUDGE_MODEL`, default `qwen/qwen3.8-27b`) from
the generator (`openai/gpt-oss-20b`) to blunt "model grades its own output" bias.

## How it works

```
                 ┌──────────────┐
inbound message ─▶│  normalise   │  mask @handles / URLs / PII, gate language
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  embed       │  sentence-transformers (local)  |  hashing fallback
                 └──────┬───────┘
             ┌──────────┼───────────────┐
             ▼          ▼               ▼
      ┌───────────┐ ┌──────────┐  ┌──────────────┐
      │ classify  │ │ retrieve │  │  signals     │
      │ logreg +  │ │ k similar│  │ sentiment,   │
      │ confidence│ │ resolved │  │ PII, risk,   │
      │           │ │ same-brnd│  │ compliance   │
      └─────┬─────┘ └────┬─────┘  └──────┬───────┘
            │            ▼               │
            │      ┌──────────┐          │
            │      │ generate │  few-shot RAG → LLMClient, + groundedness check
            │      └────┬─────┘          │
            ▼           ▼                ▼
          ┌──────────────────────────────────────┐
          │   escalation rule engine (ordered)   │
          │   → decision + reason + full trace   │
          └──────────────────────────────────────┘
```

| Stage | Choice | Why |
|---|---|---|
| Intent taxonomy | 10 intents, each with a risk tier; discovered by clustering, frozen by hand | Grounded in the data's structure; boundaries are defensible |
| Classifier | Logistic regression on frozen sentence embeddings, **weak-label** trained | Inspectable, trains in seconds, no GPU/key; hand-labels saved for eval |
| Generation | Retrieve k *resolved, same-brand* threads → few-shot an LLM; regex groundedness check | Brand voice comes from real history; hallucinated specifics get flagged |
| Escalation | Ordered rule engine over explicit signals; first match wins; whole trace returned | No second black box; every decision is auditable |
| LLM | `LLMClient` protocol; Groq / Gemini / deterministic Fake | Provider-swappable; CI and tests run fully offline |

Full detail in [`docs/design.md`](docs/design.md); the 12 non-obvious decisions
(and the alternatives rejected) are in [`docs/decisions.md`](docs/decisions.md).

## Project layout

```
src/support_agent/
  config.py            every tunable threshold, in one place
  types.py             the dataclasses passed between stages
  data/                normalise · thread the Kaggle CSV · synthetic sample
  taxonomy/            intent discovery (offline) · the frozen 10-intent taxonomy
  classify/            weak-label rules · logreg model · LLM zero-shot baseline
  generate/            resolved-reply retriever · RAG generator · verbatim baseline
  escalate/            signal extraction · the ordered rule engine · confidence-only baseline
  llm/                 LLMClient protocol · Groq · Gemini · deterministic Fake · factory
  pipeline.py          message -> Triage (intent + reply + decision + trace)
  cli.py               support-agent {demo,triage,classify,eval}
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
  real-data-notes.md   what the real 40k-thread Kaggle run showed
  labeling-guide.md    how the eval set is labelled
```

## CLI

```bash
support-agent demo
support-agent triage --brand AmazonHelp --text "Where is my order? It's been a week."
support-agent classify --text "I want to cancel my subscription"
support-agent triage --file messages.jsonl      # {"id","text","brand","history"} per line -> JSONL out
```

## Deliverables map

| Assignment ask | Here |
|---|---|
| Reproducible repo, < 15 min setup | this file · `Makefile` · `.github/workflows/ci.yml` |
| Intent categorisation | `src/support_agent/classify/`, `taxonomy/` |
| Historical-pattern response generation | `src/support_agent/generate/` |
| Auto-respond vs escalate + reasoning | `src/support_agent/escalate/` |
| Manually-labelled eval set (150–250) + sampling note | **`eval/labelset/labels.amazon.jsonl`** (200 AmazonHelp, hand-labelled) + `docs/labeling-guide.md` (§"How it is produced") |
| LLM-judge ↔ human agreement | `eval/judge_agreement.py` + `eval/judge_agreement_pairs.jsonl` (25 pairs, human-scored); result in `docs/report.md` §4 |
| Baselines: trivial + simple, per task | `docs/report.md` §3 & §4 (majority-class / canned reply / always-escalate, plus LLM zero-shot / verbatim retrieval / confidence-threshold) |
| Automated metrics + LLM quality assessment | `eval/run_*.py`, `eval/judge.py` |
| Performance vs baselines | zero-shot LLM (classification) · verbatim retrieval (generation) · confidence-threshold (escalation) — all in `docs/report.md` |
| Report: framing, 5 failure modes, metric limits, roadmap | `docs/report.md` (+ `docs/real-data-notes.md`) |
| 10–15 non-obvious decisions | `docs/decisions.md` |

## License

MIT — see [`LICENSE`](LICENSE). Dataset is CC-BY-NC-SA-4.0 (not redistributed here).
