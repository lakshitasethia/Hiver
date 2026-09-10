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

## Reproduce the headline numbers — under 15 minutes, no credentials

```bash
make setup          # venv + pip install                                (~3–5 min)
make eval-amazon    # classification + escalation on 200 hand-labelled   (~30 sec)
                    #   AmazonHelp messages — no Kaggle download, no API key
make test           # 55 tests, deterministic, offline                   (~1 min)
```

`data/amazonhelp.jsonl` (the brand slice) and `models/clf.real.joblib` (the
classifier) are committed, so `make eval-amazon` prints the report's headline
**classification** (logreg macro-F1 0.35 vs majority-class 0.06) and
**escalation** (rule-engine cost 106 vs trivial always-escalate 71) numbers with
nothing to download. Full analysis: [`docs/report.md`](docs/report.md) §4–6.

The **generation** headline (RAG beats verbatim retrieval 32–3 / 45–3 in blind
pairwise) needs a Groq key — see below. Everything (`make test`, `make eval`, CI)
runs fully offline with deterministic backends; `make demo` triages 5 example
messages end-to-end.

## Filling in the generation headline (needs a Groq key)

Classification and escalation reproduce offline (above). For the **generation**
comparison and the **LLM-judge ↔ human agreement** numbers:

```bash
cp .env.example .env                       # paste GROQ_API_KEY (free: console.groq.com/keys)

python -m eval.report --labelset eval/labelset/labels.amazon.jsonl \
       --data data/amazonhelp.jsonl --model models/clf.real.joblib --gen-limit 40
python -m eval.judge_agreement             # judge vs human on 25 pre-scored pairs
```

Groq's free tier rate-limits on tokens/minute, so `--gen-limit 40` takes
~15–40 min; that is why the committed report block shows the offline fake for
generation and quotes the real numbers (from two earlier logged runs) in §4.
The `auto` LLM backend resolves **Groq → Gemini → offline fake**; the judge runs
on a *different* family (`qwen/qwen3.8-27b`) from the generator
(`openai/gpt-oss-20b`) to blunt "model grades its own output" bias.

_Rebuilding the classifier from scratch:_ `make data && make data-amazon &&
python -m support_agent.train --data data/conversations.jsonl --out models/clf.real.joblib`
(needs a Kaggle token). Not required — `models/clf.real.joblib` is committed.

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

Full detail in [`docs/design.md`](docs/design.md); the 13 non-obvious decisions
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
  make_labelset.py     seed a label set from a dataset
  label_cli.py         keystroke-driven interactive labeller (resumable)
  run_classification.py / run_generation.py / run_escalation.py
  judge.py             LLM-as-judge (rubric + randomised pairwise)
  judge_agreement.py   judge-vs-human agreement on 25 pre-scored pairs
  tune_thresholds.py   pick escalation thresholds on a dev split
  report.py            run every suite -> docs/report.md results block
  labelset/labels.amazon.jsonl   200 hand-labelled AmazonHelp messages (the golden set)
docs/
  design.md            architecture spec
  report.md            framing · baselines · 5 failure modes · "what is misleading about
                       my headline number?" · one-more-week · attribution
  decisions.md         13 non-obvious decisions and why
  real-data-notes.md   the raw-data journey to the AmazonHelp numbers
  labeling-guide.md    how the golden set was sampled and labelled
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
| Reproducible repo; README reproduces headline results < 15 min | `make setup && make eval-amazon` (~4 min, no credentials); CI runs it |
| Pick one brand; classify · draft reply · auto-vs-escalate + reason | AmazonHelp; `classify/` + `taxonomy/` · `generate/` · `escalate/` |
| Manually-labelled eval set (150–250) + sampling note | **`eval/labelset/labels.amazon.jsonl`** (200 AmazonHelp) + `docs/labeling-guide.md` (top section) |
| Automated metrics + LLM-judge rubric + **judge↔human agreement** | `eval/run_*.py`, `eval/judge.py`, `eval/judge_agreement.py` (25 human-scored pairs; result in `docs/report.md` §4) |
| Results vs a **trivial** and a **simple** baseline, each task | `docs/report.md` §3–4: majority-class / canned reply / always-escalate  +  LLM zero-shot / verbatim retrieval / confidence-threshold |
| Report: framing + what-not-built · results vs baselines · 5 failure modes w/ examples · **"What is misleading about my headline number?"** · one-more-week · attribution | `docs/report.md` (§§1–8) |
| 10–15 non-obvious decisions | `docs/decisions.md` (13) |

## License & data

Code: MIT ([`LICENSE`](LICENSE)). `data/amazonhelp.jsonl` is a 4,054-thread
subsample of the Kaggle "Customer Support on Twitter" dataset
(`thoughtvector/customer-support-on-twitter`), redistributed here under its
**CC-BY-NC-SA-4.0** licence with attribution — the assignment explicitly permits
and encourages a subsample. `models/clf.real.joblib` is a classifier trained on
the full dataset, included so the headline reproduces without a Kaggle download.
