# Design — Transparent AI Customer-Support Agent

_Status: implemented. This document is the architectural spec the code follows._

## 1. Problem framing

Given a single inbound customer-support message (optionally with a few turns of
prior context), the agent must produce three things:

1. **Intent** — one of a small, self-defined set of categories.
2. **Reply** — a draft response in the brand's voice, grounded in how that brand
   has historically resolved similar messages.
3. **Routing decision** — `auto_send` or `escalate`, with a plain-English reason.

The assignment's own emphasis: _"the proof is worth more than the system."_ So the
design optimises for **inspectability** — every decision can be traced to a
number or a rule — over raw quality. A black-box LLM that scores 5 points higher
but cannot explain a failure is worth less here than a transparent pipeline whose
five failure modes we can name and reproduce.

### Scope decisions

| Decision | Choice | Why |
|---|---|---|
| Unit of work | One inbound message + up to N prior turns | Matches how a helpdesk actually triages; keeps eval tractable |
| Brands | All ~20 brands in the dataset | More data for the classifier; brand voice is handled per-brand at generation time via retrieval filter |
| Languages | English only | Dataset is ~English; language ID is a documented pre-filter, not a feature |
| Channels | Twitter/X public support | That is the dataset. Design notes what would change for email |
| Out of scope | Multi-step tool use, ticket CRUD, live sending | The agent drafts and routes; a human or a thin integration layer sends |

## 2. Architecture

```
                 ┌──────────────┐
inbound message ─▶│  normalise   │  strip @handles, urls, signatures; language gate
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  embed       │  sentence-transformers (local) OR hashing fallback
                 └──────┬───────┘
             ┌──────────┼───────────────┐
             ▼          ▼               ▼
      ┌───────────┐ ┌──────────┐  ┌──────────────┐
      │ classify  │ │ retrieve │  │  signals     │
      │ (logreg)  │ │ k similar│  │ sentiment,   │
      │ +conf     │ │ resolved │  │ PII, keywords│
      └─────┬─────┘ │ same-brnd│  └──────┬───────┘
            │       └────┬─────┘         │
            │            ▼               │
            │      ┌──────────┐          │
            │      │ generate │  few-shot RAG → LLMClient
            │      │  reply   │          │
            │      └────┬─────┘          │
            ▼           ▼                ▼
          ┌──────────────────────────────────┐
          │   escalation rule engine         │  transparent, ordered rules
          │   → decision + reason + signals  │
          └──────────────────────────────────┘
```

Each box is a module with a narrow interface and its own tests. The arrows are
plain Python function calls in `pipeline.py`.

## 3. Components

### 3.1 Normalisation (`data/normalize.py`)
Deterministic text cleanup: collapse whitespace, mask `@mentions` and URLs to
sentinel tokens, drop obvious signatures (`^- \w+$`, `Sent from my`). Emits a
`CleanMessage` with the raw and cleaned text so nothing is silently lost.
A langdetect-style heuristic (stopword ratio) flags non-English; those are
routed straight to `escalate` with reason `non_english`.

### 3.2 Embeddings (`embeddings.py`)
`Embedder` protocol with two implementations:
- `SentenceTransformerEmbedder` — `all-MiniLM-L6-v2`, 384-d, runs locally on the
  M-series CPU in a few ms/sentence, **no API key**.
- `HashingEmbedder` — deterministic 256-d hashed bag-of-char-ngrams. Used when
  `SUPPORT_AGENT_EMBEDDER=hashing` (CI, and anyone who wants a zero-download
  path). Quality is lower but the whole pipeline stays runnable and every test
  is deterministic.
Embeddings are cached to `.cache/emb/<sha1>.npy` keyed by `(model, text)`.

### 3.3 Intent taxonomy (`taxonomy/`)
- `discover.py` — embed a stratified sample, run K-Means for k in a range,
  pick k by silhouette score, print the top TF-IDF terms and 5 nearest messages
  per cluster. This is an **offline, human-in-the-loop** step; its output is a
  report, not a model artifact.
- `intents.py` — the frozen result of that step: 10 named intents, each with a
  one-line definition, 3 canonical examples, and a **risk tier**
  (`low` / `medium` / `high`) used by escalation. `high` = cancellation,
  billing dispute, legal/regulatory, security, outage.

Finalised intents (v1): `order_status`, `delivery_issue`, `billing_dispute`,
`cancellation`, `account_access`, `technical_bug`, `product_question`,
`complaint_feedback`, `praise_thanks`, `other_unclear`.

### 3.4 Classifier (`classify/`)
- `model.py` — `LogisticRegression` (scikit-learn, balanced class weights) on
  L2-normalised embeddings. Serialised with `joblib` to `models/clf.joblib`
  alongside a JSON sidecar recording training data hash, label counts, and
  cross-val macro-F1. Exposes `predict_with_confidence` → `(intent, prob,
  full_distribution)`. Confidence = max softmax prob; a `margin` (top1 − top2)
  is also returned because it is a better "is the model sure" signal.
- `baseline_llm.py` — zero-shot classification: the intent list + definitions in
  the prompt, `LLMClient` returns a JSON label. This is **baseline #1**.

### 3.5 Response generation (`generate/`)
- `retriever.py` — an in-memory nearest-neighbour index (`sklearn`
  `NearestNeighbors`, cosine) over **resolved** conversations, i.e. inbound
  message embedding → the agent reply that ended that thread. Filters to the
  **same brand**; falls back to cross-brand with a flag if fewer than `k_min`.
- `generator.py` — builds a few-shot prompt: brand, detected intent, 3–5
  retrieved `(customer, agent)` pairs, then the new message; instructs the
  `LLMClient` to match tone and never invent facts (order numbers, dates,
  policy specifics). Returns the draft plus the retrieved exemplars for audit.
- `baseline_retrieval.py` — **baseline #2**: skip the LLM, return the agent
  reply from the single nearest resolved conversation verbatim.

### 3.6 LLM client (`llm/`)
`LLMClient` protocol: `complete(prompt, *, system, temperature, json_mode)`.
- `GroqClient` — Groq API (OpenAI-shaped), default `openai/gpt-oss-20b`. Reads
  `GROQ_API_KEY`. The recommended backend: the free tier (~1000 req/day,
  ~8k tokens/min per model) is large enough to run the full evaluation. Honours
  the `retry-after` hint on 429s; optional `min_interval_s` throttle for long runs.
- `GeminiClient` — Google AI Studio, default `gemini-flash-lite-latest`. Reads
  `GOOGLE_API_KEY`. Works, but the free tier is ~20 req/day/model, so useful only
  for a small run. Same retry/throttle behaviour.
- `FakeLLM` — deterministic, rule-driven responses for tests, CI, and the no-key
  demo. Dispatches on the prompt (classify → JSON label, generate → templated
  reply from the exemplar, judge → rubric JSON) so it exercises real control flow.
- `make_llm()` resolves `auto` as **Groq → Gemini → fake** by which key is
  present. `make_judge_llm()` lets the LLM-judge run on a different model family
  (`SUPPORT_AGENT_JUDGE_MODEL`, e.g. `qwen/qwen3.8-27b`) from the generator, to
  reduce self-preference bias. Everything downstream is client-agnostic.

### 3.7 Escalation (`escalate/`)
- `signals.py` — pure functions producing a `Signals` dataclass:
  `clf_confidence`, `clf_margin`, `intent_risk_tier`, `sentiment` (lexicon-based
  −1..1), `anger_flag`, `retrieval_max_sim`, `pii_flags` (email/phone/card/order
  regexes), `compliance_hits` (keyword list: "chargeback", "GDPR", "lawyer",
  "cancel my account", …), `severity_hits` (safety / theft / repeated-failure /
  hard-demand regexes on the raw text — added after the real-data eval),
  `high_risk_mass` (total classifier probability on any high-risk intent),
  `is_non_english`, `history_len`.
- `rules.py` — an **ordered, transparent** rule list of 11. Each rule is
  `(name, predicate, decision, reason_template)`, first match wins:
  `non_english`, `compliance_or_legal`, `severity_cue`, `high_risk_intent`,
  `high_risk_intent_suspected` (mass ≥ 0.25), `low_model_confidence`
  (conf < 0.55 or margin < 0.10), `angry_customer_medium_risk`,
  `negative_sentiment_routine_intent` (sentiment ≤ −0.2 on a low/medium intent),
  `weak_retrieval` (sim < 0.40), `ungrounded_draft`, `pii_present_non_high` —
  else `auto_send`. Thresholds live in `config.py`, tuned on the dev split by
  `eval/tune_thresholds.py` (`--labelset` to tune on the real set). The engine
  returns every rule's truth value, so the reason is auditable, not just the
  winner.

### 3.8 Pipeline & CLI
- `pipeline.py` — `triage(message, brand, history) → Triage` bundling intent,
  reply, decision, reason, and all signals + retrieved exemplars.
- `cli.py` — `support-agent demo` (interactive, FakeLLM ok),
  `... classify -f file`, `... reply ...`, `... triage ...`, `... eval ...`.

## 4. Evaluation harness (`eval/`)

### 4.1 Label sets
- `eval/labelset/labels.amazon.jsonl` — **200 AmazonHelp messages, hand-labelled**
  (`intent`, `should_escalate`, a one-line `escalation_reason`, `split` dev/test).
  Sampled with `make_labelset --data data/amazonhelp.jsonl --unlabelled` then
  labelled row-by-row against `docs/labeling-guide.md`.
- `eval/labelset/labels.jsonl` — 200 synthetic rows with generator-truth labels;
  the CI / no-credentials path.
- `eval/judge_agreement_pairs.jsonl` — 25 `(message, reply)` pairs with **human**
  rubric scores, for the judge-agreement check.

### 4.2 Metrics
| Task | Automated | LLM-judge |
|---|---|---|
| Classification | macro-F1, per-class P/R/F1, confusion matrix, accuracy@confidence buckets | — |
| Generation | semantic similarity to historical reply, length ratio, groundedness rate (no invented order#/date/amount) | 1–5 on _helpfulness_, _tone match_, _factual caution_, _would a human send this_ (rubric in `judge.py`), pairwise vs the retrieval baseline, plus a **judge-vs-human agreement** run (`judge_agreement.py`: would_send κ, rubric MAE + Pearson r) |
| Escalation | P/R/F1 on `should_escalate`, cost-weighted error (false-auto-send ×5), auto-send rate, decision-curve sweep | — |

### 4.3 Baselines (a trivial and a simple one per task)
| Task | Trivial | Simple |
|---|---|---|
| Classification | majority class (always `delivery_issue`) | LLM zero-shot (`baseline_llm.py`) |
| Generation | one fixed canned reply for everyone | nearest resolved reply, verbatim (`baseline_retrieval.py`) |
| Escalation | always-escalate / never-escalate | confidence-threshold only, swept |

### 4.4 Report generation
`eval/report.py` runs all three suites on the label set, writes JSON to
`eval/results/` and renders the tables into `docs/report.md` between
`<!-- BEGIN:results -->` markers, so the report is reproducible, not
hand-transcribed.

## 5. Repo layout & reproducibility

- `make setup` — venv + `pip install -e ".[dev]"` (< 5 min on the target machine;
  < 15 with the model download).
- `make data` — download Kaggle CSV via `scripts/download_data.sh` (needs a
  Kaggle token), then `make data-amazon` to filter to the focus brand; or
  `make data-sample` for the committed synthetic fixture (no credentials).
- `make taxonomy` / `make train` / `make eval` / `make report`.
- Real AmazonHelp eval:
  `python -m eval.report --labelset eval/labelset/labels.amazon.jsonl --data data/amazonhelp.jsonl --model models/clf.real.joblib --gen-limit 70`
- `make demo` — one message end-to-end with `FakeLLM`.
- `make test` — pytest; runs in CI with `SUPPORT_AGENT_EMBEDDER=hashing` and
  `FakeLLM`, no network.
- Determinism: fixed seeds, pinned deps, embedding cache, `FakeLLM`.

## 6. Known limitations (expanded in the report)

- Twitter replies are short and casual; models a brand-voice that will not
  transfer to email/chat without retraining.
- "Resolved" is inferred from thread structure (agent had the last turn, no
  customer reply after) — a noisy proxy for "the customer was actually happy".
- Single-message framing loses multi-turn diagnosis context.
- Lexicon sentiment misses sarcasm — a documented failure mode.
- LLM-judge shares a family with the generator; we mitigate with a rubric and
  human spot-checks but cannot fully remove the bias.
