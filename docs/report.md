# Report — Transparent AI Support Agent for **AmazonHelp**

_Companion to [`design.md`](design.md) (architecture),
[`decisions.md`](decisions.md) (13 non-obvious choices), and
[`real-data-notes.md`](real-data-notes.md) (the raw Kaggle run — weak-label
collapse, taxonomy check, and how the real data forced the escalation rules to be
rebuilt)._

**Brand:** AmazonHelp — 4,054 threads in the working slice, 75% resolved.
Chosen because its message mix exercises every intent (delivery, order status,
refunds, account access, Prime cancellation, device/app bugs, product questions),
whereas AppleSupport (the other large brand) is almost all device bugs.

**What the results block below is.** **Classification** and **escalation** are the
real thing: evaluated on all 200 hand-labelled AmazonHelp messages, no LLM
involved except the zero-shot baseline. **Generation** rows in the block were
produced with the offline *fake* LLM — a free-tier Groq token-rate limit shut the
live generation run down mid-session — so ignore them; the **real generation
numbers** (two Groq runs, `n≈60` each, plus a judge-vs-human agreement study on
25 AmazonHelp replies) are in §4 "Reading the run" and §6, with sources. The
zero-shot classification baseline cell is likewise the fake; the real
cross-dataset figure is quoted below. **`make eval-amazon`** reproduces the
classification and escalation numbers here in ~30 s with no Kaggle download and
no API key (`data/amazonhelp.jsonl` and `models/clf.real.joblib` are committed);
generation needs a Groq key.

## 1. Problem framing: what "good" means for AmazonHelp

**Task.** For one inbound tweet to `@AmazonHelp` (plus up to 4 prior turns):
assign an intent, draft a reply in Amazon's support voice grounded in how
`@AmazonHelp` has historically resolved similar tweets, and decide `auto_send`
vs `escalate` with a reason a support lead would accept.

**"Good" for this brand, concretely:**
- **Never auto-send on money or account loss.** Refunds, unrecognised charges,
  "cancel my Prime", "someone's in my account" — a wrong automated reply here is
  expensive or irreversible. These always escalate.
- **A late-package tweet is not automatically an escalation.** Amazon's #1
  inbound is "where is my order" / "Prime said today". A calm status query can be
  auto-answered; a *failed* delivery (missing, damaged, stolen, "said delivered —
  it's not here") needs a human.
- **Non-English is common** (Spanish, French, Japanese, German in the sample) and
  is a routing target, not a feature — it always escalates.
- **The draft is a suggestion, not an outbound action.** A human or a thin
  integration layer sends. The risky verb stays outside the model.

**Framing choice — three small transparent models, not one large opaque one.**
The brief says "the proof is worth more than the system." An end-to-end LLM agent
would likely write better prose, but a single failure would be unattributable and
there would be nothing to compare it to. So the system is a pipeline of
inspectable parts — logistic-regression classification, retrieval + few-shot
generation, an ordered rule engine for routing — each individually tested and
measured against a trivial and a simple baseline.

**What I chose *not* to build:** multi-turn dialogue state, ticket/CRM writes,
live sending, a fine-tuned reply model, per-agent personalisation, and any
non-English capability beyond detecting it and escalating.

**Scope decisions and their costs:**

| Decision | Rationale | Cost accepted |
|---|---|---|
| One message + short history as the unit | Matches how triage works; keeps eval tractable | Loses multi-turn diagnosis |
| Classifier trained across all brands, **deployed for AmazonHelp** | 4k Amazon threads weak-label to too few rows/class (`product_question` = 1); the taxonomy is still defined from Amazon's clusters | The classifier is not Amazon-specialised — [decision #0](decisions.md) |
| Retrieval, brand voice, eval set, escalation tuning are **AmazonHelp-only** | This is the part that has to be brand-specific | — |
| English only, language-gated | Non-English → always escalate | No automation on ~10% of Amazon's volume |
| "Resolved" inferred from thread shape | No explicit resolution label exists | Counts "agent signed off politely" as success |
| Taxonomy frozen at 10 intents | Discovered from AmazonHelp clusters; enough for routing | Coarser than a real helpdesk's 30–50 queues |

**Data.** Kaggle "Customer Support on Twitter" (`thoughtvector/…`), threaded by
`in_response_to_tweet_id`, filtered to `author_id == "AmazonHelp"` (4,054 threads,
75% resolved). A credential-free synthetic corpus of the same shape drives CI.

## 2. Approach in one paragraph

Normalise text (mask handles/URLs/PII, gate language — CJK or a non-English
function-word lean → flag) → embed (`all-MiniLM-L6-v2`, or a hashing fallback) →
classify intent with logistic regression trained on **weak keyword labels**
(hand-labels are spent on evaluation only) → retrieve *k* similar **resolved
AmazonHelp** threads → few-shot them to an LLM (Groq `gpt-oss-20b`) for a draft,
with a post-hoc regex check that it invented no specifics → extract escalation
signals (confidence, margin, intent risk tier, sentiment, retrieval similarity,
PII, compliance keywords, safety/theft/damage/repeated-failure/money cues,
high-risk-intent probability mass) → run an **ordered 11-rule engine** that
returns a decision, a plain-English reason, and every rule's truth value.

## 3. Baselines

Each task is measured against a **trivial** and a **simple** baseline:

| Task | Ours | Simple | Trivial |
|---|---|---|---|
| Classification | logreg on embeddings | LLM zero-shot (definitions in the prompt) | majority class (always `delivery_issue`) |
| Generation | RAG few-shot | nearest resolved reply, verbatim | one fixed canned "we'll look into it" |
| Escalation | ordered rule engine | confidence-threshold only, swept | always-escalate / never-escalate |

## 4. Results

<!-- BEGIN:results -->
_Generated 2026-09-10T22:06:14 · 200 labelled examples · embedder=`all-MiniLM-L6-v2` · llm=`fake:gemini-flash-lite-latest` · judge=`fake:qwen/qwen3.8-27b`_

### Classification

| metric | logreg (ours) | LLM zero-shot (simple baseline) | majority-class (trivial → delivery_issue) |
| --- | --- | --- | --- |
| macro-F1 | **0.353** | _fake_ †  | 0.060 |
| accuracy | 0.38 | _fake_ | **0.43** |

_† zero-shot cell is the offline fake. On the earlier real cross-dataset runs the
zero-shot LLM beat logreg by ~0.12–0.15 macro-F1
([`real-data-notes.md`](real-data-notes.md) §5); expect the same or worse on Amazon._

**Per-class F1 (ours):**

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| order_status | 0.1818 | 0.75 | 0.2927 | 8 |
| delivery_issue | 0.7045 | 0.3605 | 0.4769 | 86 |
| billing_dispute | 0.5556 | 0.3333 | 0.4167 | 15 |
| cancellation | 0.375 | 1.0 | 0.5455 | 3 |
| account_access | 0.6667 | 0.2857 | 0.4 | 7 |
| technical_bug | 0.8182 | 0.4737 | 0.6 | 19 |
| product_question | 0.0 | 0.0 | 0.0 | 15 |
| complaint_feedback | 0.4483 | 0.4815 | 0.4643 | 27 |
| praise_thanks | 0.0435 | 0.0909 | 0.0588 | 11 |
| other_unclear | 0.1714 | 0.6667 | 0.2727 | 9 |

**Accuracy by confidence bucket (calibration check):**

| confidence range | n | accuracy |
| --- | --- | --- |
| 0.00-0.50 | 97 | 0.2887 |
| 0.50-0.70 | 64 | 0.4219 |
| 0.70-0.85 | 19 | 0.5789 |
| 0.85-1.01 | 20 | 0.5 |

### Generation  _(offline fake LLM — see the intro; real numbers in §4/§6)_

| metric | RAG (ours) | retrieval-only (simple baseline) | canned reply (trivial baseline) |
| --- | --- | --- | --- |
| semantic sim to historical reply | _fake_ | _fake_ | 0.35 _(real: canned is far below both)_ |
| groundedness rate | _fake_ | 1.0 (verbatim) | 1.0 (says nothing) |

_The only real signal in this fake row: the trivial **canned** reply scores ~0.35
similarity-to-history vs ~0.8 for RAG and the retrieval baseline — it is not a
contender. Real RAG-vs-retrieval numbers are two Groq runs below._

### Escalation

| metric | rule engine (ours) | confidence-only (simple) | conf-only @best sweep | always-escalate (trivial) | never-escalate (trivial) |
| --- | --- | --- | --- | --- | --- |
| f1 | 0.7829 | 0.5633 | 0.7664 | 0.7842 | 0.0 |
| false auto-sends | 10 | 60 | 6 | 0 | 129 |
| false escalations | 56 | 47 | 69 | 71 | 0 |
| weighted cost (5x/1x) | 106.0 | 347.0 | 99.0 | 71.0 | 645.0 |
| auto-send rate | 0.125 | 0.42 | 0.04 | 0.0 | 1.0 |

Cost model: a false auto-send costs 5.0x a false escalation. Auto-send rate is shown because a policy that escalates everything scores a great cost while automating nothing.
<!-- END:results -->

### Does the LLM-judge agree with a human?

Required check for any LLM-as-judge. I hand-scored **25 (message, reply) pairs**
(`eval/judge_agreement_pairs.jsonl` — real AmazonHelp replies) on the same
1–5 rubric the judge uses, then re-scored them with the judge
(`python -m eval.judge_agreement`):

| | value |
|---|---|
| `would_send` exact-match rate | **0.36** |
| `would_send` Cohen's κ (chance-corrected) | **0.07** |
| human `would_send` rate | 0.72 |
| judge (`qwen3.8-27b`) `would_send` rate | **0.08** |
| rubric-average Pearson r (human vs judge) | 0.15 |
| mean abs error — helpfulness / tone / factual-caution | 1.04 / 0.88 / 0.64 |

**The judge and a human barely agree on "would you send this" (κ ≈ 0).** The
judge is drastically stricter — it would send 8% of these replies, I would send
72%. On the graded axes it is within ~1 point of me, best on factual-caution
(0.64) and worst on helpfulness (1.04). **Consequence:** the `would_send_rate`
the judge reports on generated drafts (0.10–0.18 in the earlier Groq runs) is a
lower bound set by Qwen's strictness, not a "ready to ship" number. The
**pairwise** RAG-vs-baseline comparison is the metric to trust — both replies
face the same strict grader, so the *relative* judgment survives even when the
absolute one does not. This is picked up again in §6.

### Reading the committed run

- **Classification on real AmazonHelp is weak: logreg macro-F1 0.35, accuracy
  0.38.** The trivial majority baseline (always `delivery_issue`) gets accuracy
  **0.43** — higher, because 43% of AmazonHelp tweets *are* delivery issues — but
  macro-F1 0.06, so it is useless per-class. Logreg beats it on macro-F1 by a
  wide margin (0.35 vs 0.06) but that is a low bar. The zero-shot LLM baseline is
  the meaningful comparison and needs a working Groq run to fill in; the earlier
  cross-brand runs put it ~0.12–0.15 macro-F1 *above* logreg
  ([`real-data-notes.md`](real-data-notes.md) §5), and there is no reason to
  expect Amazon to be kinder.
- **The classifier's real problem is coverage, not the model.** `product_question`
  F1 is **0.00** (15 test rows, 0 recalled) because the cross-brand weak-labeller
  produced ~1 training row for it; `praise_thanks` F1 0.06, `delivery_issue`
  recall 0.36 (its precision is fine at 0.70 — it just misses two-thirds).
  `technical_bug` (0.60) is the only class that holds up. This is the top roadmap
  item: Amazon-specific labelling functions + a hand-labelled training seed.
- **Calibration is broken on this data — and that matters.** On the synthetic set
  confidence cleanly separated right from wrong; here the 0.85–1.0 confidence
  bucket is only **50%** accurate and the 0.7–0.85 bucket 58%. The cross-brand
  classifier is confidently wrong on AmazonHelp phrasing. This is *why* the
  escalation engine's confidence-derived signals mislead it (below), and why
  calibration is a roadmap item.
- **Escalation: the rule engine does NOT beat a trivial baseline here.** Rule
  engine weighted cost **106** (10 false auto-sends, 56 false escalations, 12.5%
  auto-send rate). **Always-escalate costs 71.** The confidence-only sweep gets
  99 by escalating 96% of everything. On this data the cost-optimal policy really
  is "send everything to a human" — see §6, this is the headline that is
  misleading. The rule engine still cut its own false-auto-sends from 20 to 10
  after the real data forced new severity/language rules
  ([`real-data-notes.md`](real-data-notes.md) §5), and it is the only option that
  *automates anything* (12.5%) while keeping misses in single digits.
- **Generation still wins its comparisons.** Real Groq runs (`n≈60`, two
  datasets) have RAG beating the verbatim-retrieval baseline in blind
  order-randomised pairwise **32–3** and **45–3** (25 / 12 ties), with
  groundedness ~0.93–0.97 and reply length close to the historical replies. The
  trivial canned reply scores ~0.35 similarity-to-history vs ~0.8 for both real
  options — not a contender. Caveat from the judge-agreement study above: trust
  the pairwise ranking, not the absolute 1–5 scores.

  | | RAG (ours) | verbatim retrieval | canned |
  |---|---|---|---|
  | blind pairwise vs verbatim (run A / run B) | **32–3** / **45–3** | — | — |
  | groundedness rate | 0.97 / 0.93 | 1.0 | 1.0 (empty) |
  | semantic sim to history | 0.60 / 0.78 | 0.66 / 0.93 | ~0.35 |
  | judge would_send (Qwen, strict) | 0.18 / 0.10 | — | — |
- **Offline / CI numbers are different again.** `make eval` with no key runs on
  the synthetic set with the fake LLM: classification macro-F1 ~0.74, fake judge
  ties every pair. That path exists for reproducibility, not for headline numbers.

## 5. Five primary failure modes

All examples are real AmazonHelp messages from the 200-row eval set, run through
the committed pipeline.

**1. The classifier collapses on Amazon's thin classes.**
`product_question` recall is **0** (15/15 missed) and `praise_thanks` F1 is 0.06,
because the cross-brand weak-labeller produced ~1 training row for
`product_question` and Amazon's praise phrasing (often non-English, or "praise
the AI overlords") does not look like the templated `praise_thanks` cues.
_Example:_ "Is Amazon Music exclusive to Echo users in India?" → predicted
`delivery_issue`. _Hypothesis:_ single shared weak-label rules cannot cover one
brand's long tail; the fix is Amazon-specific labelling functions plus a
hand-labelled training seed (roadmap #1), or a fine-tuned encoder.

**2. `delivery_issue` under-recalls (0.36) — Amazon's single most common intent.**
The classifier calls a delivery problem `order_status` or `other_unclear` when
the tweet does not contain a template-like phrase. _Example:_ "It is 8:26pm on
delivery day and no sign that [carrier] are going to make this delivery on time"
→ `order_status`, confidently (0.60), auto-sendable on intent alone. _Hypothesis:_
"where is my order" and "my order failed" share vocabulary; the weak-labeller and
the linear model both under-separate them. The escalation layer partly covers for
this with the new `theft_loss` / `damaged` severity cues, but a tweet with no
keyword ("no sign that they are going to make this") still slips.

**3. The language gate missed non-Latin scripts.**
The original `detect_english` counted English stopwords; a Japanese or
handle-plus-URL Spanish tweet has too few Latin tokens, hit the "too short to
judge → assume English" branch, and was never flagged. _Example:_ "Amazonから
届いた本がひどい状態で送られてきて、返品した…" (damaged book + gift-card shortfall)
→ classified `other_unclear`, `is_non_english=false`, auto-send. _Fix applied:_
any CJK character, or a message that leans on non-English function words, now
flags non-English; this removed ~8 of the 20 false auto-sends. _Residual:_
Latin-script languages with heavy borrowing from English can still pass.

**4. A confident-but-wrong classifier defeats the escalation rules.**
Several `billing_dispute` and `account_access` messages are classified as
`delivery_issue` (medium risk) with high confidence and strong retrieval, so no
rule fires. _Example:_ "I bought a monitor during the cyber-monday sales, and now
it's suddenly being refunded. Why?" → `delivery_issue` at 0.56 → auto-send,
though the gold is `billing_dispute` (escalate). _Fix applied:_ a `money` /
refund-phrasing severity cue and `high_risk_intent_suspected` (escalate on ≥25%
probability mass on any high-risk intent) catch some of these; a fully-calibrated
classifier (roadmap #3) is the real fix.

**5. The "resolved" proxy poisons the retrieval corpus with polite dead-ends.**
A thread ending in "Please keep us updated on the outcome" with no customer
reply counts as `resolved=true` and enters the corpus as an exemplar. The
generator then learns to stall. _Example:_ for "I desperately need help getting
back into my account", the nearest resolved AmazonHelp reply is literally "Please
keep us updated on the outcome. We're always happy to help" — which says nothing.
_Fix path:_ require a minimum agent-reply length and a closing cue ("refunded",
"reshipped", "sorted") before a thread counts as resolved; better, score
resolution with a small model.

## 6. What is misleading about my headline number?

**The single most misleading number is "the escalation rule engine is a big win."**
On the synthetic set it beat every confidence-only operating point 3–6× on
cost. On real AmazonHelp it **loses to a one-line trivial baseline**:
always-escalate costs 71, the rule engine costs 106. Three things drive that, and
none of them is "the approach is wrong":

1. **Base rate.** 129 of 200 AmazonHelp tweets genuinely need a human (money,
   account, non-English, failed delivery). When 65% should escalate, "escalate
   everything" is a strong baseline by construction.
2. **The 5:1 cost model I chose.** A false auto-send costs 5× a false escalation,
   so the 10 messages the rule engine wrongly auto-sends (×5 = 50) outweigh
   automating 25 correctly. Pick a 2:1 ratio and the ranking flips. The ratio is
   a policy input, stated in `config.py`, not a fact.
3. **The classifier is too weak (macro-F1 0.35) and miscalibrated** (see §4), so
   the rule engine's confidence- and intent-derived signals are often wrong.

So the honest read: for AmazonHelp Twitter *today*, with this classifier, the
right product is **assist the human** — draft a reply and a suggested route,
shown to an agent who always reviews — not "auto-send a confident subset". The
generation results (RAG > verbatim > canned) are what matter in that framing.
Auto-send becomes defensible once the classifier is calibrated and Amazon-tuned
(roadmap #1, #3).

**Other ways the numbers mislead:**

- **The judge's `would_send` rate is not a quality score.** §4 shows the judge
  agrees with a human on "would you send this" at κ ≈ 0 (it sends 8%, I send
  72%). Only the **pairwise** RAG-vs-baseline ranking survives; the absolute 1–5
  and would_send figures are lower bounds set by Qwen's strictness.
- **Semantic-similarity-to-history rewards mimicry.** The verbatim baseline
  scores *higher* on similarity (~0.9 vs ~0.8) yet loses the blind pairwise
  45–3. Copying past phrasing ≠ being the right reply for this customer.
- **`macro-F1` hides the class that matters.** Overall 0.35, but `delivery_issue`
  — 43% of real volume — has recall 0.36. A number that averages the rare
  classes with the common one understates how often the agent misroutes the
  tweets AmazonHelp actually gets.
- **Escalation gold is one annotator's policy.** `should_escalate` was labelled
  by me against a fixed rubric; a second annotator would disagree on the
  borderline "angry vent vs actionable complaint" and "product question vs
  needs-authority" calls. Inter-annotator agreement is unmeasured (the labelling
  guide's self-diff pass is a weak proxy).
- **No latency/cost SLA.** logreg + MiniLM is ~1 ms/message; a Groq generation
  call is ~0.5–5 s and free-tier token-rate-limited (this is why generation is a
  small subset). Production would batch, cache retrievals, and use a cheaper
  judge.

## 7. What I'd do next with one more week

**Days**
1. **Fix the classifier's recall on the intents that matter for Amazon.**
   `delivery_issue` vs `order_status` and `billing_dispute` recall are the weak
   points (§5); the fastest win is more weak-labelling functions for Amazon
   phrasing ("Prime said today", "said delivered — nothing here", "still no
   refund") plus a ~100-row hand-labelled training seed.
2. **Fix failure mode #3** — normalise `#`/punctuation before the groundedness
   containment check; removes needless escalations at zero risk.
3. **Calibrate the classifier probabilities** (isotonic on a held-out slice) so
   `low_confidence` means the same thing across intents — this is what shrinks
   the escalation engine's false-escalation count (§5).
4. **Grow the eval set to Amazon's full intent mix.** `cancellation` (3 rows),
   `account_access` (7) and `order_status` (8) are too thin for their per-class
   numbers to be trusted; label ~150 more, stratified.

**Weeks**
5. Replace logreg with a fine-tuned `MiniLM`/`DeBERTa-small` classifier; keep
   logreg as the transparent baseline and ship whichever wins per class.
6. Tighten "resolved" (min agent-reply length + a closing cue like "refunded" /
   "reshipped") and rebuild the AmazonHelp retrieval corpus so the generator
   stops learning from polite dead-ends.
7. A trained sentiment/sarcasm signal for escalation, and a small **panel** of
   judges (different providers) reporting agreement, to further de-bias §4.
8. Brand-canonical sign-off injection so cross-thread retrieval never imports the
   wrong voice.

**Longer**
9. Multi-turn context: condition classification and generation on the whole
   thread, not just the first customer message.
10. Human-in-the-loop feedback: log every escalation outcome (did the human
    agree?) and every auto-reply's downstream signal (customer replied angrily?
    re-opened?) and retrain thresholds + classifier on it.

## 8. Attribution

- **Dataset:** Kaggle "Customer Support on Twitter"
  (`thoughtvector/customer-support-on-twitter`), CC-BY-NC-SA-4.0. Not
  redistributed; downloaded at build time.
- **Libraries:** `sentence-transformers` (`all-MiniLM-L6-v2` embeddings),
  `scikit-learn` (`LogisticRegression`, `NearestNeighbors`, metrics), `numpy`,
  `pandas`, `groq` and `google-genai` SDKs, `rich` (CLI).
- **Models:** Groq-hosted `openai/gpt-oss-20b` (generation, zero-shot baseline)
  and `qwen/qwen3.8-27b` (LLM-judge); Google `gemini-flash-lite-latest` supported
  as an alternate backend.
- **Ideas borrowed:** weak supervision / labelling-functions is the Snorkel
  pattern (Ratner et al., 2017), hand-rolled here to keep dependencies light.
  RAG few-shot from retrieved exemplars is the standard retrieval-augmented
  generation recipe. LLM-as-judge with a fixed rubric and position-randomised
  pairwise follows common practice (e.g. MT-Bench / Zheng et al., 2023).
- **AI assistance:** built with an AI coding assistant (Claude). All design
  decisions, the taxonomy, the hand labels, the judge scores, and this report
  are mine and defensible.
