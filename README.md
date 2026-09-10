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
