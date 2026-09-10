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
`LLMClient` protocol: `complete(prompt, *, system, temperature, response_schema)`.
- `GeminiClient` — Google AI Studio, `gemini-1.5-flash`, free tier. Reads
  `GOOGLE_API_KEY`. Retries with backoff on 429/5xx; hard timeout.
- `FakeLLM` — deterministic, rule-driven responses for tests and for the
  no-key demo. Keyed off the prompt so classification/judge tests are stable.
Selected by `make_llm()` from env; everything downstream is client-agnostic.

### 3.7 Escalation (`escalate/`)
- `signals.py` — pure functions producing a `Signals` dataclass:
  `clf_confidence`, `clf_margin`, `intent_risk_tier`, `sentiment` (lexicon-based
  −1..1), `anger_flag`, `retrieval_max_sim`, `pii_flags` (email/phone/card/order
  regexes), `compliance_hits` (keyword list: "chargeback", "GDPR", "lawyer",
  "cancel my account", …), `is_non_english`, `history_len`.
- `rules.py` — an **ordered, transparent** rule list. Each rule is
  `(name, predicate, decision, reason_template)`. First match wins. Examples:
  `non_english → escalate`, `compliance_hit → escalate`,
  `high_risk_intent → escalate`, `low_confidence (conf<0.55 or margin<0.15) →
  escalate`, `angry_and_medium_risk → escalate`,
  `weak_retrieval (max_sim<0.45) → escalate`, else `auto_send`.
  Thresholds live in `config.py` and are tuned on the dev split, not guessed.
  The engine returns every rule's truth value so the reason is auditable, not
  just the winning one.

### 3.8 Pipeline & CLI
- `pipeline.py` — `triage(message, brand, history) → Triage` bundling intent,
  reply, decision, reason, and all signals + retrieved exemplars.
- `cli.py` — `support-agent demo` (interactive, FakeLLM ok),
  `... classify -f file`, `... reply ...`, `... triage ...`, `... eval ...`.

## 4. Evaluation harness (`eval/`)

### 4.1 Label set
`eval/labelset/labels.jsonl` — 150–250 examples sampled **stratified by
predicted cluster and brand** from a held-out slice the classifier never trains
on. Each row: `id, brand, text, history, intent (gold), ideal_reply_notes,
should_escalate (gold), escalation_reason (gold), notes`. Labelled by hand with
`docs/labeling-guide.md`; every ambiguous call is recorded in `notes`.

### 4.2 Metrics
| Task | Automated | LLM-judge |
|---|---|---|
| Classification | macro-F1, per-class P/R/F1, confusion matrix, accuracy@confidence buckets | — |
| Generation | semantic similarity to historical reply, length ratio, groundedness heuristics (no invented order#/date), refusal rate | 1–5 on _helpfulness_, _tone match_, _factual caution_, _would a human send this_ (rubric in `judge.py`), pairwise vs retrieval baseline |
| Escalation | P/R/F1 on `should_escalate`, cost-weighted error (false-auto-send penalised 5×), decision-curve vs confidence-only | — |

### 4.3 Baselines
1. **Classification**: LLM zero-shot (`baseline_llm.py`).
2. **Generation**: nearest-resolved-reply verbatim (`baseline_retrieval.py`).
3. **Escalation**: confidence-threshold-only (single number, no rules).
Every headline number is reported **against its baseline**, with the delta.

### 4.4 Report generation
`eval/report.py` runs all three suites on the label set, writes JSON to
`eval/results/` and renders the tables into `docs/report.md` between
`<!-- BEGIN:results -->` markers, so the report is reproducible, not
hand-transcribed.

## 5. Repo layout & reproducibility

- `make setup` — venv + `pip install -e ".[dev]"` (< 5 min on the target machine;
  < 15 with the model download).
- `make data` — download Kaggle CSV via `scripts/download_data.sh` (needs
  `~/.kaggle/kaggle.json`); or `make data-sample` to use the committed
  `data/sample/` fixture (2k rows, redistributable slice) so the pipeline runs
  with **no credentials at all**.
- `make taxonomy` / `make train` / `make eval` / `make report`.
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
